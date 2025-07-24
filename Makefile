.PHONY: mypy
mypy:
	mypy --config mypy.ini self_* sandbox_runner
