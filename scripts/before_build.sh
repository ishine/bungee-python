#!/bin/bash
set -e
echo "Updating Git submodules..."
# Assuming the script is in 'scripts' dir, go to project root
cd "$(dirname "$0")/.."
git submodule update --init --recursive
echo "Submodules updated. before-build script finished."