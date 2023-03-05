# Symbolic Regression TDC Demo

## Install dependencies (insider project root folder)
```
$ sudo pip3 install poetry
$ sudo poetry install
```

- Install [Julia](https://www.digitalocean.com/community/tutorials/how-to-install-julia-programming-language-on-ubuntu-22-04) (necessary for PySR);
- Download and install [TuringBot for Linux](https://turingbotsoftware.com/download.html). Tested only on Linux;


How to run the examples:
```
poetry run python3 -m examples.simple_pysr
poetry run python3 -m examples.simple_turingbot
```

## Testing
```
poetry run coverage run -m pytest -s -v tests
poetry run coverage report
poetry run coverage xml
```