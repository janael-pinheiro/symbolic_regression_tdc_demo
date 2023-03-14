# Symbolic Regression TDC Demo

This repository contains code for demo integration with TuringBot, PySR and Gplearn tools. Therefore, this code is highly experimental. Contributions are more than welcome.

## Install dependencies (insider project root folder)
- Install [Julia](https://www.digitalocean.com/community/tutorials/how-to-install-julia-programming-language-on-ubuntu-22-04) (necessary for PySR);
- Download and install [TuringBot for Linux](https://turingbotsoftware.com/download.html). Tested only on Linux;

```
$ sudo pip3 install poetry
$ sudo poetry install --no-root
$ sudo poetry run python3 -c "import pysr; pysr.install()"
```


How to run the examples:
```
poetry run python3 -m examples.simple_pysr
poetry run python3 -m examples.simple_turingbot
```

## Command Line Interface

You can interact with TuringBot, PySR and Gplearn tools through the CLI demonstrated below (inside the root folder of the project):
```
$ poetry run python3 -m sr.cli --help
$ poetry run python3 -m sr.cli --features-filepath Datasets/Titanic/proceed/train_experiment_features.csv --target-filepath Datasets/Titanic/proceed/survived.csv --algorithm turingbot
$ poetry run python3 -m sr.cli --features-filepath Datasets/Titanic/proceed/train_experiment_features.csv --target-filepath Datasets/Titanic/proceed/survived.csv --algorithm gplearn
```

## Datasets
The datasets used in the experiments come from [Kaggle](https://www.kaggle.com/competitions/titanic/data).


## Testing
```
poetry run coverage run -m pytest -s -v tests
poetry run coverage report
poetry run coverage xml
```