#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${root_dir}/build"

cmake -S "${root_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${build_dir}" --parallel

"${build_dir}/lab01_lifetime"
"${build_dir}/lab02_pool_reuse"
"${build_dir}/lab03_pool_ipc"
"${build_dir}/lab04_p2p_optional"

