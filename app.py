import streamlit as st
from transformers import pipeline
import pandas as pd

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="IMDb Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide"
)

# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown("""
<style>

.stApp{
    background: linear-gradient(135deg,#050816,#0f172a,#09090b);
    color:white;
}

/* Main Title */
.main-title{
    font-size:48px;
    font-weight:700;
    color:white;
    margin-bottom:5px;
}

.sub-title{
    font-size:18px;
    color:#cbd5e1;
    margin-bottom:25px;
}

/* Cards */
.card{
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.08);
    padding:25px;
    border-radius:20px;
    margin-top:20px;
}

/* Positive Card */
.positive-card{
    background:#0f2d1f;
    padding:20px;
    border-left:5px solid #22c55e;
    border-radius:15px;
}

/* Negative Card */
.negative-card{
    background:#34131a;
    padding:20px;
    border-left:5px solid #ef4444;
    border-radius:15px;
}

/* Info Cards */
.info-card{
    background:rgba(255,255,255,0.04);
    border:1px solid rgba(255,255,255,0.06);
    border-radius:20px;
    padding:20px;
    min-height:180px;
}

/* Buttons */
.stButton > button{
    width:100%;
    background:linear-gradient(
        90deg,
        #7c3aed,
        #9333ea
    );
    color:white;
    border:none;
    border-radius:12px;
    height:55px;
    font-size:18px;
    font-weight:600;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#161b2d;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ==================================================
# MODEL
# ==================================================

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

model = load_model()

# ==================================================
# SESSION STATE
# ==================================================

if "history" not in st.session_state:
    st.session_state.history = []

# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("""
    # 🎬 IMDb Movie Review

    ### Sentiment Analyzer
    """)

    st.markdown("---")

    page = st.radio(
        "Navigation",
        [
            "🏠 Analyze Review",
            "ℹ️ About",
            "❓ How It Works"
        ]
    )

    st.markdown("---")

    st.markdown("""
    ### 👨‍💻 Developer

    **Shaukat Aziz**

    NLP & Machine Learning Project
    """)

# ==================================================
# ANALYZE PAGE
# ==================================================

def show_analyzer():

    st.markdown(
        '<div class="main-title">Analyze a Movie Review</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sub-title">Enter a movie review below and we will predict whether it is positive or negative.</div>',
        unsafe_allow_html=True
    )

    review = st.text_area(
        "",
        height=220,
        placeholder="e.g. This movie was amazing..."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Characters", len(review))

    with col2:
        st.metric("Words", len(review.split()))

    analyze = st.button("✨ Analyze Sentiment")

    if analyze:

        if not review.strip():
            st.warning("Please enter a movie review.")
            return

        result = model(review)[0]

        label = result["label"]
        score = result["score"] * 100

        st.markdown("## Analysis Result")

        if label == "POSITIVE":

            st.markdown(f"""
            <div class="positive-card">
            <h2>😊 Positive Review</h2>
            <h3>Confidence: {score:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(score))

        else:

            st.markdown(f"""
            <div class="negative-card">
            <h2>😞 Negative Review</h2>
            <h3>Confidence: {score:.2f}%</h3>
            </div>
            """, unsafe_allow_html=True)

            st.progress(int(score))

        # Save history
        st.session_state.history.append({
            "Review": review[:80],
            "Prediction": label,
            "Confidence": f"{score:.2f}%"
        })

        # Download report
        report = f"""
IMDb Movie Review Sentiment Analysis Report

Developer:
Shaukat Aziz

Review:
{review}

Prediction:
{label}

Confidence:
{score:.2f}%
"""

        st.download_button(
            "📄 Download Analysis Report",
            report,
            file_name="sentiment_report.txt",
            mime="text/plain"
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # History Section
    if st.session_state.history:

        st.markdown("## 📊 Prediction History")

        history_df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            history_df,
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="info-card">
        <h3>😊 Positive</h3>
        <p>Reviews expressing satisfaction and positive emotions.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-card">
        <h3>😞 Negative</h3>
        <p>Reviews expressing dissatisfaction and negative emotions.</p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-card">
        <h3>🤖 AI Powered</h3>
        <p>Powered by DistilBERT Transformer NLP model.</p>
        </div>
        """, unsafe_allow_html=True)

# ==================================================
# ABOUT PAGE
# ==================================================

def show_about():

    st.title("ℹ️ About This Project")

    st.markdown("""
    ## IMDb Movie Review Sentiment Analyzer

    This project uses Natural Language Processing (NLP)
    and Machine Learning techniques to determine whether
    a movie review is Positive or Negative.

    ### Technologies Used

    - Python
    - Streamlit
    - Hugging Face Transformers
    - DistilBERT
    - PyTorch

    ### Key Features

    - Real-time Sentiment Prediction
    - Transformer-based AI Model
    - Modern User Interface
    - Downloadable Reports
    - Prediction History Tracking

    ### Model Information

    DistilBERT is a lightweight version of BERT that
    provides fast and accurate sentiment analysis.

    ### Developed By

    **Shaukat Aziz**
    """)

# ==================================================
# HOW IT WORKS PAGE
# ==================================================

def show_how_it_works():

    st.title("❓ How It Works")

    st.markdown("""
    ## Sentiment Analysis Workflow

    ### Step 1
    User enters a movie review.

    ### Step 2
    The review text is processed by the DistilBERT model.

    ### Step 3
    Natural Language Processing techniques extract meaning.

    ### Step 4
    The AI predicts whether the review is:

    - Positive
    - Negative

    ### Step 5
    Confidence score is displayed to the user.

    ### Example

    Review:

    > "This movie was absolutely amazing and I loved every minute."

    Prediction:

    > Positive

    Confidence:

    > 99%

    ---
    The model was trained on sentiment classification tasks
    and can understand context, emotions, and opinions.
    """)

# ==================================================
# ROUTING
# ==================================================

if page == "🏠 Analyze Review":
    show_analyzer()

elif page == "ℹ️ About":
    show_about()

elif page == "❓ How It Works":
    show_how_it_works()

# ==================================================
# FOOTER
# ==================================================

st.markdown("---")

st.markdown(
    """
    <center>
    <h4>🎬 IMDb Movie Review Sentiment Analyzer</h4>
    <p>Developed by <b>Shaukat Aziz</b></p>
    <p>Powered by DistilBERT • Streamlit • Hugging Face</p>
    </center>
    """,
    unsafe_allow_html=True
)
