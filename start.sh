#!/bin/bash
# Download the spaCy model if not already installed
python -m spacy download en_core_web_sm
# Start the Flask server
python server.py
