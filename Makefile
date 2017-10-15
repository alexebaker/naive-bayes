all: docs

.PHONY: clean
clean:
	find . -name "*.pyc" -delete
	find . -name "*.npz" -delete

.PHONY: docs
docs:
	sphinx-apidoc -f -o docs/source/ naive_bayes/ && pushd docs && make html && popd


