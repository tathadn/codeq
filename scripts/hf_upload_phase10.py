"""Phase 10: upload Round 2 LoRA adapter + preference datasets to the Hub."""
from huggingface_hub import HfApi, create_repo

MODEL_REPO = "tathadn/codeq-qwen2.5-coder-7b-dpo-r2"
DATA_REPO = "tathadn/codeq-debugbench-dpo-pairs"
MODEL_DIR = "/home/lamassunobackup/tdebnath/codeqB/models/agentq-round2"
DATA_DIR = "/home/lamassunobackup/tdebnath/codeqB/data/preferences"

api = HfApi()
print("whoami:", api.whoami()["name"])

create_repo(MODEL_REPO, repo_type="model", exist_ok=True)
create_repo(DATA_REPO, repo_type="dataset", exist_ok=True)

print(f"Uploading adapter -> {MODEL_REPO}")
api.upload_folder(
    folder_path=MODEL_DIR,
    repo_id=MODEL_REPO,
    repo_type="model",
    ignore_patterns=["ref/*", "ref/**"],
    commit_message="Phase 10: upload Round 2 DPO LoRA adapter + model card",
)

print(f"Uploading preferences -> {DATA_REPO}")
api.upload_folder(
    folder_path=DATA_DIR,
    repo_id=DATA_REPO,
    repo_type="dataset",
    commit_message="Phase 10: upload Round 1/2 DPO preference pairs + dataset card",
)

print("done")
print(f"  https://huggingface.co/{MODEL_REPO}")
print(f"  https://huggingface.co/datasets/{DATA_REPO}")
