#!/bin/bash
apt install -y tesseract-ocr tesseract-ocr-por
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -c "import nltk; nltk.download('rslp'); nltk.download('stopwords'); nltk.download('punkt');"