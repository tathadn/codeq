#!/usr/bin/env bash
# Receive trained LoRA adapters from Machine B.
# Usage: bash scripts/sync_from_b.sh ROUND MACHINE_B_HOST [MACHINE_B_USER]
#
# Example:
#   bash scripts/sync_from_b.sh 1 192.168.1.42 user

set -euo pipefail

ROUND=${1:?Usage: sync_from_b.sh ROUND MACHINE_B_HOST [USER]}
MACHINE_B_HOST=${2:?Usage: sync_from_b.sh ROUND MACHINE_B_HOST [USER]}
MACHINE_B_USER=${3:-user}
ADAPTER_DIR="models/agentq-round${ROUND}"
REMOTE="${MACHINE_B_USER}@${MACHINE_B_HOST}"

mkdir -p "${ADAPTER_DIR}"

echo "Receiving LoRA adapters from ${REMOTE} -> ${ADAPTER_DIR}/"
scp -r "${REMOTE}:~/codeq/models/agentq-round${ROUND}/adapter_*" "${ADAPTER_DIR}/"
scp "${REMOTE}:~/codeq/models/agentq-round${ROUND}/adapter_config.json" "${ADAPTER_DIR}/"
echo "Adapters received in ${ADAPTER_DIR}/"
