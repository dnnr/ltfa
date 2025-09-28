# 📈 ltfa

ltfa is short for _long-term financial analysis_.
It is a tool that aims to provide a comprehensive overview for personal finances with a focus on investment returns.
In order to compensate for a highly fragmented accounts landscape, it is able to process bank statements in many forms and consolidates them into a single ledger.

## Features
- Customizable CSV parsing
- Detecting and tracking inter-account transactions
- Tracking salary, savings, and capital gains with rolling averages to expose short- and long-term trends
- Interactive Bokeh dashboard

## Installation
- Run `pipenv sync` to prepare virtualenv with dependencies
- Activate the environment with `pipenv shell` or prefix commands with `pipenv run`.

## Usage
- Maintain a configuration file (defaults to `~/.config/ltfa/ltfa.yaml`).
- Example commands:
  - `pipenv run python ltfa.py --config path/to/config.yaml -B ltfa.html`
  - `pipenv run python ltfa.py --config path/to/config.yaml -I investment-report.txt`

## Configuration
There is currently no example configuration available, but test scenarios provide a rough idea.
