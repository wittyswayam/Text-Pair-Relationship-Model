# Natural Language Inference (NLI) using Machine Learning

This project implements a **Text Pair Relationship Detection System** using classical Machine Learning techniques.
The model takes **two sentences** as input and predicts the semantic relationship between them:

* **Entailment**
* **Contradiction**

The system is built using the **SNLI (Stanford Natural Language Inference)** dataset and includes complete preprocessing, model training, evaluation, and a simple inference function.

---

## 🔍 Project Overview

Natural Language Inference (NLI) is a core NLP task where a model determines how two sentences are logically related.
This project:

* Loads and preprocesses the SNLI dataset
* Cleans text data (stopwords, punctuation, lowercasing)
* Combines paired sentences into a single input
* Converts text to vectors using **TF-IDF**
* Trains an **ML classifier** (Logistic Regression / SVM / similar depending on your notebook)
* Builds an end-to-end prediction function `detect()`
* Saves the trained model and vectorizer for future use

---

## 📂 Dataset

The project uses the **SNLI 1.0** dataset.
From the dataset, only three useful fields were kept:

* `sentence1`
* `sentence2`
* `label1`

Neutral samples were removed to keep only two classes:
✔ entailment
✔ contradiction

For training speed, a sample of **5000 rows** was used.

---

## 🛠️ Workflow

### 1. Data Cleaning

* Lowercasing
* Stopword removal
* Removing unwanted labels
* Joining sentence pairs into a single combined input

### 2. Train-Test Split

The dataset is split into **train (80%)** and **test (20%)**.

### 3. Text Vectorization

A **TF-IDF vectorizer** converts the cleaned text into numerical vectors.

### 4. Model Training

A machine learning classifier is trained on vectorized sentence pairs to learn semantic relationships.

### 5. Evaluation

Accuracy and basic metrics are reviewed on the test dataset.

---

## 🚀 Inference: Sentence Relationship Detection

The notebook includes a convenient function:

```python
def detect(inp1, inp2, model, tfidf):
    # Returns: "entailment" or "contradiction"
```

Example:

```python
inp1 = "A man is playing a guitar."
inp2 = "A man is riding a horse."
prediction = detect(inp1, inp2, model, tfidf)
print(prediction)   # Expected: contradiction
```

---

## 💾 Saving the Model

Both the trained model and TF-IDF vectorizer are saved:

```python
pickle.dump(model, open("models/model.pkl", "wb"))
pickle.dump(tfidf, open("models/tfidf.pkl", "wb"))
```

This allows easy reuse without retraining.

---

## 📁 Repository Structure

```
├── models/
│   ├── model.pkl         # trained ML model
│   └── tfidf.pkl         # TF-IDF vectorizer
├── notebook.ipynb        # main project code
├── README.md
```

---

## 📌 Future Improvements

* Use deep learning models like BERT or RoBERTa
* Add support for the "neutral" class
* Deploy as a simple API
* Build a Streamlit UI for sentence pair testing

---
