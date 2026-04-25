#!/usr/bin/env python3
"""
Person 3 (hours 20–22): upload a saved model directory to the Hugging Face Hub.

Auth: set HF_TOKEN (recommended) or run `huggingface-cli login` first.

Example:
  python scripts/push_model_to_hub.py --folder ./sqlsage-grpo/merged --repo-id your-org/sqlsage-trained
"""

from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Upload a local model folder to Hugging Face Hub.")
    parser.add_argument("--folder", required=True, help="Directory containing config.json, tokenizer, weights")
    parser.add_argument("--repo-id", required=True, help="HF repo id: org-or-user/model-name")
    parser.add_argument("--private", action="store_true", help="Create/use a private repo")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        print("Set HF_TOKEN or run `huggingface-cli login` before pushing.", file=sys.stderr)
        return 1

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("pip install huggingface_hub", file=sys.stderr)
        return 1

    api = HfApi(token=token)
    create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True, token=token)
    api.upload_folder(
        folder_path=args.folder,
        repo_id=args.repo_id,
        repo_type="model",
        token=token,
    )
    print(f"Uploaded {args.folder} -> https://huggingface.co/{args.repo_id}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
