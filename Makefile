PYTHON ?= python3

.PHONY: smoke compile api push-hf

compile:
	$(PYTHON) -m compileall sqlsage

smoke:
	$(PYTHON) scripts/smoke_env.py

api:
	uvicorn sqlsage.app:app --reload --port 8000

# Fast Space update: Git push (delta). See README "Hugging Face Space (deploy)".
# Uses scripts/push_hf_fast.sh (HF_TOKEN for headless, else push via hf-space remote).
push-hf:
	bash scripts/push_hf_fast.sh
