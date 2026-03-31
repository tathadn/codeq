"""DPO training script — NOT USED ON MACHINE A.

This file is present for completeness and so that Machine B can run it
after pulling code from git. Do not execute on this machine.

Machine B runs:
    python -m src.train_dpo \\
        --data data/preferences/round1.jsonl \\
        --base models/qwen2.5-coder-7b \\
        --output models/agentq-round1
"""

raise RuntimeError(
    "train_dpo.py must not be run on Machine A. "
    "Copy code to Machine B and run it there."
)
