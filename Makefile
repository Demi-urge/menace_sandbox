.PHONY: mypy synergy-graph install-self-improvement-deps check-context-builder self-coding-check
mypy:
	mypy --config mypy.ini self_* sandbox_runner
	python scripts/check_governed_embeddings.py
	python scripts/check_governed_retrieval.py
	python scripts/check_context_builder_usage.py
	python tools/check_self_coding_registration.py

synergy-graph:
	python module_synergy_grapher.py --build

install-self-improvement-deps:
	python scripts/install_self_improvement_deps.py

check-context-builder:
	python scripts/check_context_builder_usage.py

self-coding-check:
	python tools/check_self_coding_usage.py
	python tools/find_unmanaged_bots.py
	python tools/check_coding_bot_decorators.py
