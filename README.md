# 🎬 IMDb Sentiment Analysis (Transformer Edition)

A high-performance sentiment analysis tool that moves beyond traditional statistical word-counting by utilizing **DistilBERT**, a state-of-the-art Transformer model. This project transforms raw text into accurate sentiment predictions (Positive/Negative) using deep learning context-awareness.

## - Features
* **Context-Aware:** Unlike older TF-IDF/Logistic Regression models, this engine understands negation ("not good") and complex sarcasm.
* **Minimalist Architecture:** No massive CSV files or pickle dependencies required; the model downloads pre-trained weights automatically.
* **Modern Pipeline:** Uses Hugging Face `transformers` to deliver enterprise-grade performance in a simple, portable script.

## - Prerequisites
You need Python 3.8+ installed. 

Install the required libraries:
```bash
pip install transformers torch streamlit
