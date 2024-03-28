default: linter tests

install:
	pip install -U pip
	pip install -U -e '.[dev]'

linter:
	flake8 torchclust && mypy torchclust
	flake8 tests && mypy tests

tests:
	coverage run -m pytest tests
	coverage report

api_docs:
	pdoc3 --html -o api_docs -f torchclust

dist:
	python setup.py sdist

clean:

.PHONY: linter tests api_docs dist