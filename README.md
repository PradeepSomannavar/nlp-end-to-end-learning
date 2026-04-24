# NLP End-to-End Learning

This repository contains tutorials and projects for Natural Language Processing (NLP).

## Contents

- **Text Preprocessing**: A comprehensive guide to text cleaning, tokenization, lemmatization, and stopword removal.
- **SMS Spam Classification**: A complete project implementing spam detection using:
  - Bag of Words (BoW)
  - TF-IDF
  - Word2Vec and Average Word2Vec

## Setup

1. Install dependencies:
   ```bash
   pip install nltk re
   ```
2. Download NLTK data:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```