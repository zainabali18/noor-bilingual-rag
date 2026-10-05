.PHONY: install ingest run eval test clean

install:
	pip install -r requirements.txt

ingest:
	python -m noor.ingest

run:
	python -m streamlit run app.py

eval:
	python -m eval.evaluate

test:
	pytest -q

clean:
	rm -rf .chroma __pycache__ .pytest_cache
	find . -name "__pycache__" -type d -prune -exec rm -rf {} +
