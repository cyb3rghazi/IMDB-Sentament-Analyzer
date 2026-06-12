import streamlit as st
from transformers import pipeline

# 1. Page Config
st.set_page_config(page_title="AI Sentiment Analysis", page_icon="🎬")

# 2. Load Model (with cache for performance)
@st.cache_resource
def load_model():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

model = load_model()

# 3. UI Layout
st.title("🎬 IMDb Sentiment Analyzer")
st.write("A deep-learning powered tool using **DistilBERT** to analyze movie review sentiment.")

user_input = st.text_area("Enter your movie review here:", height=150)

if st.button("Analyze Sentiment"):
    if user_input.strip():
        with st.spinner("Analyzing context..."):
            result = model(user_input)[0]
            label = result['label']
            confidence = result['score'] * 100
            
            # Display result with nice colors
            if label == "POSITIVE":
                st.success(f"### Result: {label}")
            else:
                st.error(f"### Result: {label}")
            
            st.progress(result['score'])
            st.write(f"Confidence: **{confidence:.2f}%**")
    else:
        st.warning("Please enter some text to analyze.")
