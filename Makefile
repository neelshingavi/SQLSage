PYTHON ?= python3

.PHONY: smoke compile api

compile:
	$(PYTHON) -m compileall sqlsage

smoke:
	$(PYTHON) scripts/smoke_env.py

api:
	uvicorn sqlsage.app:app --reload --port 8000
