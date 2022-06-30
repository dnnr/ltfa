#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

hadolint /src/Dockerfile
find /src -name '*.sh' -exec shellcheck {} \+

cd /src
mypy --ignore-missing-imports ltfa
pytest -x
