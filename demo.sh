#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
core_dir="${BOSS_DEMO_CORE_DIR:-${repo_dir}/../build/deps}"
transaction_engine="${BOSS_TRANSACTION_ENGINE:-${repo_dir}/build/libBOSSTransactionLogEngine.so}"
storage_engine="${BOSS_STORAGE_ENGINE:-${repo_dir}/../BOSSInMemoryWriteEngine/build/libBOSSInMemoryWriteEngine.so}"

cmake -S "${repo_dir}" -B "${repo_dir}/build" \
  -DBUILD_BOSS_DEMO=ON -DBOSS_DEMO_CORE_DIR="${core_dir}"
cmake --build "${repo_dir}/build" --target BOSSDemo

if [[ ! -f "${transaction_engine}" || ! -f "${storage_engine}" ]]; then
  echo "Engine library not found. Build both engines or set" >&2
  echo "BOSS_TRANSACTION_ENGINE and BOSS_STORAGE_ENGINE." >&2
  exit 2
fi

exec "${repo_dir}/build/BOSSDemo" \
  --transaction-engine "${transaction_engine}" \
  --storage-engine "${storage_engine}" "$@"
