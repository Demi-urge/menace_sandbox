#!/bin/bash
set -e
python -m pip install --upgrade pip
pip install -e .
pip install pytest
pip install stripe python-dotenv scikit-learn
pip install hypothesis
