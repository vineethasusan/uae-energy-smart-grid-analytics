export PYTHONPATH := src

.PHONY: install pipeline dashboard test lint docker

install:    ## install dependencies
	pip install -r requirements-dev.txt
pipeline:   ## load + solar forecasts and inspection list -> outputs/report.md
	python -m energy.pipeline
dashboard:  ## http://localhost:8501
	streamlit run dashboard.py
test:
	pytest -q
lint:
	ruff check .
docker:
	docker build -t uae-energy-smart-grid-analytics . && docker run -p 8501:8501 uae-energy-smart-grid-analytics
