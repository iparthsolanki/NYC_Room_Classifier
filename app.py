import streamlit as st
import pandas as pd
import joblib
import time
import os

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Room Classifier | ML Portfolio",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "Model_Pipeline.pkl")

COLUMNS = [
    "latitude",
    "longitude",
    "price",
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
    "availability_365",
    "neighbourhood_group",
    "neighbourhood",
]

NEIGHBOURHOOD_GROUPS = ["Manhattan", "Brooklyn", "Queens", "Bronx", "Staten Island"]

ROOM_ICONS = {
    "Entire home/apt": "🏡",
    "Private room": "🛏️",
    "Shared room": "🛋️",
}

# ----------------------------------------------------------------------------
# CUSTOM CSS
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 15% 15%, #1b1140 0%, #0d0b26 35%, #060714 70%),
                    linear-gradient(135deg, #0f0c29 0%, #1a1a4a 40%, #0d2b3e 100%);
        background-attachment: fixed;
        color: #e9e9f5;
    }

    #MainMenu, footer, header {visibility: hidden;}

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #12122e 0%, #0b0b21 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }

    .hero-wrap {
        text-align: center;
        padding: 2.2rem 1rem 1.2rem 1rem;
    }

    .badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(99,102,241,0.18), rgba(45,212,191,0.18));
        border: 1px solid rgba(129,140,248,0.35);
        color: #a5b4fc;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        margin-bottom: 1.1rem;
    }

    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 800;
        font-size: 3.2rem;
        line-height: 1.1;
        background: linear-gradient(90deg, #818cf8 0%, #38bdf8 45%, #2dd4bf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
    }

    .hero-subtitle {
        font-size: 1.08rem;
        color: #b6b8d8;
        font-weight: 400;
        margin-top: 0.6rem;
        max-width: 620px;
        margin-left: auto;
        margin-right: auto;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.045);
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 20px;
        padding: 1.8rem 1.9rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.35);
        margin-bottom: 1.3rem;
    }

    .section-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.15rem;
        color: #d8d9f6;
        margin-bottom: 0.9rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    div[data-testid="stNumberInput"] input,
    div[data-testid="stTextInput"] input,
    div[data-baseweb="select"] > div {
        background-color: rgba(255,255,255,0.055) !important;
        border: 1px solid rgba(255,255,255,0.13) !important;
        border-radius: 12px !important;
        color: #f1f1fb !important;
    }

    label {
        color: #b7b9de !important;
        font-weight: 500 !important;
        font-size: 0.86rem !important;
    }

    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #6366f1 0%, #06b6d4 100%);
        color: white;
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.85rem 0;
        border-radius: 14px;
        border: none;
        box-shadow: 0 6px 24px rgba(99,102,241,0.45);
        transition: all 0.25s ease;
        letter-spacing: 0.02em;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 10px 32px rgba(6,182,212,0.5);
        color: white;
    }

    .result-card {
        background: linear-gradient(135deg, rgba(99,102,241,0.16), rgba(6,182,212,0.10));
        border: 1px solid rgba(129,140,248,0.4);
        border-radius: 22px;
        padding: 2rem;
        text-align: center;
        animation: fadeIn 0.6s ease;
        margin-bottom: 1.2rem;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .result-icon { font-size: 3rem; margin-bottom: 0.4rem; }

    .result-label {
        font-family: 'Poppins', sans-serif;
        font-size: 1.9rem;
        font-weight: 800;
        color: #ffffff;
        margin: 0.2rem 0;
    }

    .confidence-chip {
        display: inline-block;
        margin-top: 0.7rem;
        padding: 5px 16px;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        letter-spacing: 0.03em;
    }

    .conf-high { background: rgba(52,211,153,0.18); color: #34d399; border: 1px solid rgba(52,211,153,0.4); }
    .conf-medium { background: rgba(251,191,36,0.18); color: #fbbf24; border: 1px solid rgba(251,191,36,0.4); }
    .conf-low { background: rgba(248,113,113,0.18); color: #f87171; border: 1px solid rgba(248,113,113,0.4); }

    .prob-row {
        margin-bottom: 0.85rem;
    }

    .prob-label {
        display: flex;
        justify-content: space-between;
        font-size: 0.88rem;
        color: #cfd0ef;
        margin-bottom: 4px;
        font-weight: 500;
    }

    .prob-track {
        width: 100%;
        height: 12px;
        background: rgba(255,255,255,0.07);
        border-radius: 999px;
        overflow: hidden;
    }

    .prob-fill {
        height: 100%;
        border-radius: 999px;
        background: linear-gradient(90deg, #6366f1, #06b6d4);
        animation: growBar 1s ease forwards;
    }

    .prob-fill.best {
        background: linear-gradient(90deg, #34d399, #06b6d4);
    }

    @keyframes growBar {
        from { width: 0%; }
    }

    .footer-sig {
        text-align: center;
        color: #7d7fa3;
        font-size: 0.85rem;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid rgba(255,255,255,0.08);
    }

    .sidebar-heading {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 1.05rem;
        color: #a5b4fc;
        margin-top: 1.3rem;
        margin-bottom: 0.5rem;
    }

    .sidebar-text {
        font-size: 0.86rem;
        color: #b6b8d8;
        line-height: 1.55;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(MODEL_PATH)


model_load_error = None
model = None
try:
    model = load_model()
except Exception as e:
    model_load_error = str(e)

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🏠 Room Classifier")
    st.markdown(
        '<div class="sidebar-text">A machine learning app that predicts the room '
        "category of an Airbnb-style listing from its location, pricing, and "
        "activity signals.</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">🧠 Model</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">Random Forest Classifier trained inside a '
        "scikit-learn Pipeline, with median imputation and standard scaling for "
        "numeric features, and most-frequent imputation with one-hot encoding "
        "for categorical features.</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">📊 Features Used</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="sidebar-text">
        • Latitude &amp; Longitude<br>
        • Price per night<br>
        • Minimum nights<br>
        • Number of reviews<br>
        • Reviews per month<br>
        • Host listings count<br>
        • Availability (365 days)<br>
        • Neighbourhood group<br>
        • Neighbourhood
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sidebar-heading">⚙️ Tech Stack</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-text">Python • scikit-learn • pandas • joblib • '
        "Streamlit • FastAPI-compatible schema</div>",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if model_load_error:
        st.error("Model failed to load. See main panel for details.")
    else:
        st.success("Model loaded and ready ✅")

# ----------------------------------------------------------------------------
# HERO SECTION
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-wrap">
        <div class="badge">Machine Learning • Streamlit • FastAPI Compatible</div>
        <div class="hero-title">Room Classifier</div>
        <div class="hero-subtitle">Predict the room category using Airbnb-style listing features</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model_load_error:
    st.error(
        f"⚠️ Could not load `Model_Pipeline.pkl`. Make sure the file sits in the "
        f"same folder as `app.py`.\n\nDetails: {model_load_error}"
    )
    st.stop()

# ----------------------------------------------------------------------------
# INPUT FORM
# ----------------------------------------------------------------------------
left_col, right_col = st.columns(2, gap="large")

with left_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📍 Location &amp; Pricing</div>', unsafe_allow_html=True)

    latitude = st.number_input(
        "Latitude", min_value=-90.0, max_value=90.0, value=40.75000, step=0.00001, format="%.5f"
    )
    longitude = st.number_input(
        "Longitude", min_value=-180.0, max_value=180.0, value=-73.98000, step=0.00001, format="%.5f"
    )
    price = st.number_input("Price ($ per night)", min_value=0.01, value=150.0, step=1.0)
    minimum_nights = st.number_input("Minimum Nights", min_value=1, max_value=365, value=3, step=1)
    number_of_reviews = st.number_input("Number of Reviews", min_value=0, value=25, step=1)

    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Activity &amp; Host Details</div>', unsafe_allow_html=True)

    reviews_per_month = st.number_input(
        "Reviews per Month", min_value=0.0, value=1.2, step=0.1, format="%.2f"
    )
    calculated_host_listings_count = st.number_input(
        "Host Listings Count", min_value=0, value=2, step=1
    )
    availability_365 = st.number_input(
        "Availability (365)", min_value=0, max_value=365, value=180, step=1
    )
    neighbourhood_group = st.selectbox("Neighbourhood Group", NEIGHBOURHOOD_GROUPS, index=0)
    neighbourhood = st.text_input("Neighbourhood", value="Midtown")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🔮 Predict Room Type", use_container_width=True)

# ----------------------------------------------------------------------------
# PREDICTION
# ----------------------------------------------------------------------------
if predict_clicked:
    errors = []
    if not neighbourhood.strip():
        errors.append("Neighbourhood cannot be empty.")
    if price <= 0:
        errors.append("Price must be greater than 0.")

    if errors:
        for err in errors:
            st.error(f"⚠️ {err}")
    else:
        with st.spinner("Analyzing listing features..."):
            time.sleep(0.6)
            try:
                input_data = {
                    "latitude": latitude,
                    "longitude": longitude,
                    "price": price,
                    "minimum_nights": int(minimum_nights),
                    "number_of_reviews": int(number_of_reviews),
                    "reviews_per_month": reviews_per_month,
                    "calculated_host_listings_count": int(calculated_host_listings_count),
                    "availability_365": int(availability_365),
                    "neighbourhood_group": neighbourhood_group,
                    "neighbourhood": neighbourhood.strip(),
                }
                row = pd.DataFrame([input_data])[COLUMNS]

                prediction = model.predict(row)[0]
                probabilities = model.predict_proba(row)[0]
                classes = model.classes_

                prob_pairs = sorted(
                    zip(classes, probabilities), key=lambda x: x[1], reverse=True
                )
                top_class, top_prob = prob_pairs[0]

                if top_prob >= 0.70:
                    conf_label, conf_class = "High Confidence", "conf-high"
                elif top_prob >= 0.45:
                    conf_label, conf_class = "Medium Confidence", "conf-medium"
                else:
                    conf_label, conf_class = "Low Confidence", "conf-low"

                icon = ROOM_ICONS.get(prediction, "🏘️")

                st.markdown(
                    f"""
                    <div class="result-card">
                        <div class="result-icon">{icon}</div>
                        <div class="sidebar-text" style="font-size:0.95rem; letter-spacing:0.05em; text-transform:uppercase; color:#9fa1cf;">Predicted Room Type</div>
                        <div class="result-label">{prediction}</div>
                        <span class="confidence-chip {conf_class}">✔ {conf_label} — {top_prob*100:.1f}%</span>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(
                    '<div class="section-title">📊 Prediction Probabilities</div>',
                    unsafe_allow_html=True,
                )

                for cls, prob in prob_pairs:
                    is_best = cls == top_class
                    bar_class = "prob-fill best" if is_best else "prob-fill"
                    pct = prob * 100
                    star = " ⭐" if is_best else ""
                    st.markdown(
                        f"""
                        <div class="prob-row">
                            <div class="prob-label"><span>{ROOM_ICONS.get(cls, '🏘️')} {cls}{star}</span><span>{pct:.1f}%</span></div>
                            <div class="prob-track"><div class="{bar_class}" style="width:{pct}%;"></div></div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"⚠️ Prediction failed: {e}")

# ----------------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-sig">
        Built with Streamlit &amp; scikit-learn · Room Classifier ML App · Portfolio Project
    </div>
    """,
    unsafe_allow_html=True,
)
