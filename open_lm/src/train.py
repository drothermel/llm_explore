import itertools
import logging
import math
import time
from contextlib import nullcontext

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import ReduceOp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

try:
    import wandb
except ImportError:
    wandb = None

from open_lm.data import sample_chunk
from open_lm.distributed import is_master
from open_lm.precision import get_autocast
from open_lm.meters import AverageMeter


def unwrap_model(model):
    if hasattr(model, "module"):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def train_one_epoch(
    model,
    data,
    loss,
    epoch,
    step,
    optimizer,
    scaler,
    scheduler,
    total_steps,
    args,
    tb_writer=None,
):
    """Trains model for one epoch on the provided data.

    Returns:
        success (bool): Whether training completed successfully
        step (int): Global step at the end of the epoch. Note that "epoch" actually is not one full pass through the
            data, but rather the number of tokens specified by `--train-num-samples`, rounded based on shard size.
            As such, the number of steps in an "epoch" can vary, and we have to keep track of steps separately.
    """
    device = torch.device(args.device)
    autocast = get_autocast(args.precision)

    model.train()

    data["train"].set_epoch(
        epoch
    )  # set epoch in process safe manner via sampler or shared_epoch
    dataloader = data["train"].dataloader
    num_batches_per_epoch = dataloader.num_batches
    sample_digits = math.ceil(math.log(dataloader.num_samples + 1, 10))

    losses_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    forward_time_m = AverageMeter()
    backward_time_m = AverageMeter()
    optim_step_time_m = AverageMeter()
    sync_time_m = AverageMeter()

    # used only if --log-logit-mean flag is passed
    logit_m = AverageMeter()

    end = time.time()

    data_iterator = iter(dataloader)

    for i in itertools.count():
        if not args.skip_scheduler:
            scheduler(step)

        if step >= total_steps:
            logging.warning(
                f"step: {step} has reached/exceeded total_steps: {total_steps}. ending training."
            )
            break

        try:
            batch = next(data_iterator)
            has_data = torch.tensor(1, dtype=torch.long, device=device)
        except StopIteration:
            has_data = torch.tensor(0, dtype=torch.long, device=device)

        if args.world_size > 1:
            dist.all_reduce(has_data, op=ReduceOp.SUM)
        if has_data < args.world_size:
            break

        (texts,) = batch
        texts = torch.LongTensor(texts).to(device)
        data_time_m.update(time.time() - end)
        optimizer.zero_grad()

        if args.accum_freq == 1:
            with autocast():
                forward_start = time.time()
                inputs, targets = sample_chunk(texts, args)
                out, _, _ = model(inputs)
                forward_time_m.update(time.time() - forward_start)

                if args.log_logit_mean:
                    logit_m.update(torch.mean(out).item())

                total_loss = loss(out.reshape(-1, args.vocab_size), targets.reshape(-1))

            backward_start = time.time()
            backward(total_loss, scaler)
            backward_time_m.update(time.time() - backward_start)

        else:
            # split up batch into accum_freq chunks -- if you have --batch-size 8 and --accum-freq 4
            # then you only process 2 items at a time. batch-size must be divisible by accume-freq.
            assert (
                args.per_gpu_batch_size % args.accum_freq == 0
            ), "Per-GPU batch size must be divisible by accum_freq"
            per_batch = args.per_gpu_batch_size // args.accum_freq

            inputs, targets = sample_chunk(texts, args)

            forward_total_time = 0
            backward_total_time = 0
            for ii in range(args.accum_freq):
                maybe_no_sync = nullcontext
                # Don't sync gradients until the final batch for FSDP.
                if isinstance(model, FSDP) and ii != args.accum_freq - 1:
                    maybe_no_sync = model.no_sync
                with maybe_no_sync():
                    with autocast():
                        forward_start = time.time()
                        inputs_ii = inputs[ii * per_batch : (ii + 1) * per_batch]
                        if inputs_ii.shape[0] == 0:
                            break
                        targets_ii = targets[ii * per_batch : (ii + 1) * per_batch]
                        out, _, _ = model(inputs_ii)
                        forward_total_time += time.time() - forward_start

                        if args.log_logit_mean:
                            logit_m.update(torch.mean(out).item())

                        local_loss = (
                            loss(
                                out.reshape(-1, args.vocab_size), targets_ii.reshape(-1)
                            )
                            * inputs_ii.shape[0]
                            / inputs.shape[0]
                        )

                    backward_start = time.time()
                    backward(local_loss, scaler)
                    backward_total_time += time.time() - backward_start
                if ii == 0:
                    total_loss = local_loss
                else:
                    total_loss = local_loss

            forward_time_m.update(forward_total_time)
            backward_time_m.update(backward_total_time)

        optim_step_start = time.time()
        if scaler is not None:
            if args.grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip_norm, norm_type=2.0
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip_norm is not None:
                if isinstance(model, FSDP):
                    model.clip_grad_norm_(args.grad_clip_norm, norm_type=2.0)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.grad_clip_norm, norm_type=2.0
                    )
            optimizer.step()
        optim_step_time_m.update(time.time() - optim_step_start)

        global_loss_tensor = total_loss.detach().clone()

        sync_start = time.time()
        if args.world_size > 1:
            dist.all_reduce(global_loss_tensor, op=ReduceOp.AVG)
        sync_time_m.update(time.time() - sync_start)

        batch_time_m.update(time.time() - end)
        end = time.time()

        batch_count = i + 1
        step += 1
        if is_master(args):
            batch_size = len(inputs)
            # update the loss meter with the global loss tensor every iteration, so that the logging is of the avg of loss of the last
            # args.log_every_n_steps iterations
            losses_m.update(global_loss_tensor.item(), batch_size)
            if (
                i % args.log_every_n_steps == 0
                or batch_count == num_batches_per_epoch
                or step == total_steps - 1
            ):
                num_samples = batch_count * batch_size * args.world_size
                samples_per_epoch = dataloader.num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_epoch

                # gathered_loss = [torch.zeros_like(total_loss) for _ in range(args.world_size)]
                # torch.distributed.all_gather(gathered_loss, total_loss)

                # losses_m.update(sum(gathered_loss).item() / args.world_size, batch_size * args.world_size)
                losses_m.update(global_loss_tensor.item(), batch_size)
                samples_per_second = inputs.numel() * args.world_size / batch_time_m.val
                samples_per_second_per_gpu = inputs.numel() / batch_time_m.val
                loss_str = f"Loss: {losses_m.avg:.3f}"
                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"{loss_str} "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
                    f"LR: {optimizer.param_groups[0]['lr']:5f} "
                )

                # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
                log_data = {
                    "loss": losses_m.val,
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "forward_time": forward_time_m.val,
                    "backward_time": backward_time_m.val,
                    "optim_step_time": optim_step_time_m.val,
                    "sync_time": sync_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr": optimizer.param_groups[0]["lr"],
                    "tokens": (step + 1) * args.global_batch_size * args.seq_len,
                    "expected_steps_epoch": data["train"].dataloader.num_batches,
                    "seen_steps_epoch": batch_count,
                }

                if args.log_logit_mean:
                    log_data["logit_mean"] = logit_m.val

                for name, val in log_data.items():
                    name = "train/" + name
                    if tb_writer is not None:
                        tb_writer.add_scalar(name, val, step)
                    if args.wandb:
                        assert wandb is not None, "Please install wandb."
                        wandb.log(
                            {name: val, "step": step, "tokens": log_data["tokens"]}
                        )

                # resetting batch / data time meters per log window
                batch_time_m.reset()
                data_time_m.reset()
                forward_time_m.reset()
                backward_time_m.reset()
                optim_step_time_m.reset()
                sync_time_m.reset()

                if math.isnan(losses_m.val):
                    # case where loss goes to nan, we see this sometimes with bad nodes.
                    # in this case we would like to free resources and prevent other issues
                    # e.g., saving checkpoints and optmization states that may lead to skipped
                    # training on restarts.
                    return False, step

                # reset all average meters
                losses_m.reset()

    # end for
    if tb_writer is not None:
        tb_writer.flush()
    return True, step
