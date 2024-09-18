# open-lm

Currently setup with `rye`.

## Install and Setup Rye

Follow instructions from the [rye project](https://rye.astral.sh/):
```shell
# Install rye
curl -sSf https://rye.astral.sh/get | bash

# Add the following to your .bashrc equivalent (after conda path updates!)
source "$HOME/.rye/env"
```

## Using Rye

Setup rye by:
```
# Cloning this repo

cd llm_explore/open_lm/
ls # kick off rye setup of python env

# Verify your shims are properly setup
python -c "import sys; print(sys.prefix)"
#    this should print the path to the venv from rye

# Have rye make the venv for this repo
rye sync
```

Key commands:
```
# Show state of env
rye show

# Show deps
rye list

# Add a dependency
rye add <pip_package_name>

# Add a dev dependency
rye add <pip_package_name> --dev

# Remove a dependency
rye remove <pip_package_name> (optional: --dev)

# Refresh State (mainly happens automatically
rye sync

# Run black
rye fmt

# Run lint (and fix easy things)
rye lint --fix

# Run tests
rye test -v -s

# Run program
rye run <tool_name>

# Pin a version of python
rye pin 3.12
```

And learn to run scripts by seeing the bottom of [this page](https://rye.astral.sh/guide/basics/#inspecting-the-project).
