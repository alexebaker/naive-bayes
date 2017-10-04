# Naive Bayes

UNM CS 429/529 Machine Learning Project 2: Naive Bayes


## Details

Details about this project can be found on [Kaggle](https://inclass.kaggle.com/c/cs529-project2)


## Usage

**NOTE**: This code will work with either python 2 or python 3.

The main entry point for this project is `nb.py`. Use the `-h` flag from any command to see help:

## Documentation

This module uses documentation complied by [sphinx](http://www.sphinx-doc.org/en/stable/) located in the `docs/` directory. First, Shpinx needs to be installed into a virtual env:

First, you need to initialize the virtualenv:

```bash
virtualenv .venv
```

Next, activate the virtualenv in your current shell:

```bash
source .venv/bin/activate
```

Now, install the python requirements:

```bash
pip install -r requirements.txt
```

You can deactivate the virtualenv with the following command, however, make sure the virtualenv is active when you build the documentation:

```bash
deactivate
```

Now you can build the documentation. To build the documentation, run the Makefile:

```bash
source .venv/bin/activate
make docs
```

Once the documentation is built, it can be viewed in your brower by running the `open-docs.py` script:

```bash
python open-docs.py
```


## TODO

- [ ] - Implement Naive Bayes


## Authors

* [Alexander Baker](mailto:alexebaker@unm.edu)

* [Caleb Waters](mailto:waterscaleb@unm.edu)

* [Mark Mitchell](mailto:mamitchell@unm.edu)
