PYTHON := .venv/bin/python

.PHONY: install test smoke train-random evaluate-random build-submission verify-submission build-archive

install:
	uv sync --python 3.12 --extra dev

test:
	uv run --python 3.12 pytest

smoke:
	bash scripts/smoke_test.sh

train-random:
	uv run --python 3.12 mountaincar-train --config configs/smoke/random.yaml --overwrite

evaluate-random:
	uv run --python 3.12 mountaincar-evaluate --config configs/smoke/random.yaml --split validation --overwrite

build-submission:
	uv run --python 3.12 python scripts/build_submission.py

verify-submission:
	@uv run --python 3.12 python scripts/run_submission_verification.py

build-archive:
	uv run --python 3.12 python scripts/build_release_archive.py
