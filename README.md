# NLP End-to-End Learning

This repository serves as a comprehensive collection of my learnings, tutorials, and projects focused on Natural Language Processing (NLP). It tracks my journey from basic text preprocessing techniques to building end-to-end machine learning models for text classification.

## 📚 Concepts Covered

The repository is structured to progressively build NLP skills, covering the following fundamental and advanced concepts:

### 1. Text Preprocessing (`textpreprocessing.ipynb`)
This notebook demonstrates the essential steps required to clean and prepare raw text data for machine learning models.
- **Tokenization**: Breaking down paragraphs into sentences using `nltk.sent_tokenize` and sentences into words using `nltk.word_tokenize`.
- **Text Cleaning**: Utilizing Regular Expressions (`re`) to remove punctuation and numbers, and standardizing text to lowercase.
- **Stopwords Removal**: Filtering out common English words that do not add significant meaning using NLTK's `stopwords` corpus.
- **Word Normalization**:
  - **Stemming**: Using `PorterStemmer` to reduce words to their root forms.
  - **Lemmatization**: Using `WordNetLemmatizer` to reduce words to their meaningful base dictionary forms.

### 2. Feature Engineering & Embeddings
Converting text into numerical formats that machine learning models can understand:
- **Bag of Words (BoW)**: Representing text based on word occurrence using `CountVectorizer`.
- **TF-IDF (Term Frequency-Inverse Document Frequency)**: Weighing the importance of words in a document relative to a corpus using `TfidfVectorizer`.
- **Word2Vec**: Training custom dense word embeddings from scratch using `gensim.models.Word2Vec`.
- **Average Word2Vec**: Aggregating Word2Vec word embeddings to create a single vector representation for an entire sentence or document.

### 3. Deep Learning for NLP (Sequence Models)
Advanced deep learning architectures tailored for sequence data and language modeling:
- **Recurrent Neural Networks (RNNs)**: Understanding sequential data processing and capturing temporal dynamics in text.
- **Long Short-Term Memory (LSTM)**: Implementing memory cells to retain long-term dependencies and context, effectively mitigating the vanishing gradient problem.
- **Gated Recurrent Units (GRU)**: Utilizing a simplified, computationally efficient gating mechanism for sequence learning.
- **Transformers & Attention Mechanisms**: Exploring self-attention and state-of-the-art transformer architectures (e.g., BERT, GPT) for advanced language understanding and generation.


---

## 🚀 Projects

### SMS Spam Classification Project
**Path:** `NLP_Project/nlp_spam_bow_tfidf_word2vec_avgword2vec.ipynb`

An end-to-end machine learning project aimed at classifying SMS messages as either "Spam" or "Ham" (Not Spam). 

**Workflow & Implementations:**
1. **Dataset**: Used the popular SMS Spam Collection dataset.
2. **Data Cleaning Pipeline**: Applied lowercasing, regex cleaning, stopwords removal, and stemming/lemmatization to the entire dataset.
3. **Model Iterations**:
   - Trained a **Multinomial Naive Bayes (`MultinomialNB`)** model using **Bag of Words** features.
   - Trained a **Multinomial Naive Bayes (`MultinomialNB`)** model using **TF-IDF** features (including bigrams).
   - Evaluated a **Random Forest Classifier** alongside Naive Bayes for comparative performance.
4. **Deep Learning Preparations**:
   - Tokenized sentences and trained a custom **Word2Vec** model on the specific SMS corpus using `Gensim`.
   - Explored integrating pre-trained Google News vectors (`word2vec-google-news-300`).
5. **Evaluation**: Used `accuracy_score` and comprehensive `classification_report` (Precision, Recall, F1-Score) from `scikit-learn` to measure the effectiveness of the models.

---

## ⚙️ Setup and Installation

To run the notebooks in this repository, you'll need to install the required Python libraries and download the necessary NLTK datasets.

1. **Install Python dependencies:**
   ```bash
   pip install pandas scikit-learn nltk gensim tqdm
   ```

2. **Download required NLTK data:**
   Open a Python console or Jupyter cell and run:
   ```python
   import nltk
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   ```
