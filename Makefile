.PHONY: install index serve serve-a2a test lint demo clean

install:
	bash scripts/setup.sh

index:
	python -m pyrag index ./docs ./data/samples

serve:
	python -m pyrag serve

serve-a2a:
	python -m pyrag a2a serve

test:
	pytest -q

lint:
	ruff check pyrag tests

demo:
	python -m pyrag demo

clean:
	rm -rf data/chroma data/cache data/sessions .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +