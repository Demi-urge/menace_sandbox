.PHONY: mypy synergy-graph
mypy:
	mypy --config mypy.ini self_* sandbox_runner
	python scripts/check_governed_embeddings.py
	python scripts/check_governed_retrieval.py

synergy-graph:
	python module_synergy_grapher.py --build
