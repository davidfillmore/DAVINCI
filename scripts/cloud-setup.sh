#!/usr/bin/env bash
#
# Cloud environment setup for Claude Code on the web (claude.ai/code).
#
# The cloud sandbox ships Python/pip/uv but NOT conda, and DAVINCI's test suite
# needs the `davinci` conda env (cartopy, pyhdf, monet/monetio, numba),
# which is painful to install via plain pip. This script builds that env once;
# the cloud environment caches the result, so subsequent sessions start ready.
#
# Point your cloud environment's setup-script field at this file.
#
# IMPORTANT: conda `activate` state does NOT persist across an agent's separate
# tool calls. Invoke the interpreter by full path instead, e.g.:
#     $HOME/miniconda/envs/davinci/bin/pytest
#     $HOME/miniconda/envs/davinci/bin/mypy davinci_monet

set -euo pipefail

CONDA_DIR="${CONDA_DIR:-$HOME/miniconda}"
ENV_NAME="davinci"
ENV_PY="$CONDA_DIR/envs/$ENV_NAME/bin/python"

# 1. Install Miniconda if not already present.
if [ ! -x "$CONDA_DIR/bin/conda" ]; then
    echo "==> Installing Miniconda into $CONDA_DIR"
    curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
    rm -f /tmp/miniconda.sh
else
    echo "==> Miniconda already present at $CONDA_DIR"
fi

# 2. Create (or update) the davinci env from environment.yml.
if "$CONDA_DIR/bin/conda" env list | grep -qE "^\s*${ENV_NAME}\s"; then
    echo "==> Updating existing '$ENV_NAME' env from environment.yml"
    "$CONDA_DIR/bin/conda" env update -n "$ENV_NAME" -f environment.yml --prune
else
    echo "==> Creating '$ENV_NAME' env from environment.yml"
    "$CONDA_DIR/bin/conda" env create -f environment.yml
fi

# 3. Install the package in development mode with dev extras.
echo "==> Installing davinci_monet in editable mode with [dev] extras"
"$CONDA_DIR/envs/$ENV_NAME/bin/pip" install -e ".[dev]"

# 4. Smoke-check the toolchain.
echo "==> Verifying toolchain"
"$ENV_PY" -c "import davinci_monet; print('davinci_monet import OK')"
"$CONDA_DIR/envs/$ENV_NAME/bin/pytest" --version
"$CONDA_DIR/envs/$ENV_NAME/bin/mypy" --version

echo "==> Setup complete. Run tests with:"
echo "    $CONDA_DIR/envs/$ENV_NAME/bin/pytest"
