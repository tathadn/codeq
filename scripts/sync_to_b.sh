#!/usr/bin/env bash
# Send preference data to Machine B for DPO training.
# Usage: bash scripts/sync_to_b.sh ROUND MACHINE_B_HOST [MACHINE_B_USER]
#
# Example:
#   bash scripts/sync_to_b.sh 1 192.168.1.42 user

set -euo pipefail

ROUND=${1:?Usage: sync_to_b.sh ROUND MACHINE_B_HOST [USER]}
MACHINE_B_HOST=${2:?Usage: sync_to_b.sh ROUND MACHINE_B_HOST [USER]}
MACHINE_B_USER=${3:-user}
PREF_FILE="data/preferences/round${ROUND}.jsonl"
REMOTE="${MACHINE_B_USER}@${MACHINE_B_HOST}"

if [[ ! -f "${PREF_FILE}" ]]; then
    echo "ERROR: ${PREF_FILE} not found. Run preference construction first."
    exit 1
fi

echo "Syncing ${PREF_FILE} -> ${REMOTE}:~/codeq/data/preferences/"
scp "${PREF_FILE}" "${REMOTE}:~/codeq/data/preferences/"
echo "Done."
