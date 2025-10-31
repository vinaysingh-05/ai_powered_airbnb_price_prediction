import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import time
import requests
import matplotlib.pyplot as plt

# ================================
# PAGE SETTINGS
# ================================
st.set_page_config(
    page_title="AI-Powered Airbnb Price Dashboard",
    page_icon="🏙️",
    layout="wide"
)

# ================================
# FIXED NAVBAR
# ================================
st.markdown("""
<style>
    /* Remove default padding/margins from Streamlit */
    .block-container {padding-top: 0rem;}
    header {visibility: hidden;}

    /* Navbar Styling */
    .navbar {
        background-color: #ffffff;
        padding: 15px 50px;
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 9999;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .nav-title {
        font-weight: 800;
        font-size: 20px;
        color: #1C2833;
    }
    .nav-links {
        display: flex;
        gap: 25px;
    }
    .nav-links a {
        text-decoration: none;
        color: #1C2833;
        font-weight: 600;
        font-size: 16px;
        transition: color 0.3s ease;
    }
    .nav-links a:hover {
        color: #3498DB;
    }
</style>

<!-- NAVBAR HTML -->
<div class="navbar">
    <div class="nav-title">🏙️ AI-Powered Airbnb</div>
    <div class="nav-links">
        <a href="#features">Features</a>
        <a href="#get-started">Get Started</a>
        <a href="#project-intro">Project Intro</a>
        <a href="#contact">Contact</a>
    </div>
</div>
""", unsafe_allow_html=True)

# Add space below navbar
st.markdown("<br><br><br><br>", unsafe_allow_html=True)

# ================================
# LOTTIE ANIMATION
# ================================
def load_lottie_url(url):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except:
        return None

try:
    from streamlit_lottie import st_lottie
    lottie_airbnb = load_lottie_url("https://assets9.lottiefiles.com/packages/lf20_3rwasyjy.json")
except:
    lottie_airbnb = None

# ================================
# CUSTOM CSS THEME
# ================================
st.markdown("""
<style>
    .main-title {
        font-size: 46px;
        text-align: center;
        color: #1C2833;
        font-weight: 800;
    }
    .sub-title {
        text-align: center;
        font-size: 20px;
        color: #5D6D7E;
        margin-bottom: 40px;
    }
    .footer {
        text-align: center;
        color: #5D6D7E;
        font-size: 14px;
        margin-top: 80px;
    }
    .footer a {
        color: #3498DB;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
</style>
""", unsafe_allow_html=True)

# ================================
# HERO SECTION
# ================================
st.markdown("<div id='features'></div>", unsafe_allow_html=True)

col1, col2 = st.columns([1.5, 1])
with col1:
    st.markdown("<div class='main-title'>AI-Powered Airbnb Price Prediction Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Analyze, visualize, and predict real-world Airbnb pricing using Machine Learning</div>", unsafe_allow_html=True)
with col2:
    if lottie_airbnb:
        st_lottie(lottie_airbnb, height=280, key="airbnb")
    else:
        st.image("https://cdn-icons-png.flaticon.com/512/869/869869.png", width=180)

st.markdown("---")

# ================================
# LOAD DATA
# ================================
@st.cache_data
def load_data():
    return pd.read_csv("bnb_housingdb.csv")

df = load_data()

# ================================
# DATA PREPROCESSING
# ================================
if 'price' not in df.columns:
    st.error("❌ 'price' column missing in dataset.")
    st.stop()

df = df.dropna(subset=['price'])
label_encoders = {}
categorical_columns = ['room_type', 'minimum_nights', 'number_of_reviews']

for col in categorical_columns:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# ================================
# MODEL TRAINING
# ================================
X_cols = ['room_type', 'minimum_nights', 'number_of_reviews', 'minimum_nights', 'availability_365']
X = df[X_cols]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# ================================
# PREDICTION SECTION
# ================================
st.markdown("<div id='get-started'></div>", unsafe_allow_html=True)
st.header("🏘️ Get Started – Make Your Prediction")

col1, col2, col3 = st.columns(3)
with col1:
    room_type = st.selectbox("Room Type", label_encoders['room_type'].classes_)
with col2:
    neighbourhood_group = st.selectbox("minimum nights", label_encoders['minimum_nights'].classes_)
with col3:
    neighbourhood = st.selectbox("number of reviews", label_encoders['number_of_reviews'].classes_)

col4, col5 = st.columns(2)
with col4:
    minimum_nights = st.slider("Minimum Nights", 1, 365, 3)
with col5:
    availability_365 = st.slider("Availability (days per year)", 0, 365, 180)

# Encode input
encoded_input = np.array([
    label_encoders['room_type'].transform([room_type])[0],
    label_encoders['minimum_nights'].transform([neighbourhood_group])[0],
    label_encoders['number_of_reviews'].transform([neighbourhood])[0],
    minimum_nights,
    availability_365
]).reshape(1, -1)

if st.button("✨ Predict Now"):
    with st.spinner("Running AI model..."):
        time.sleep(1.2)
        predicted_price = model.predict(encoded_input)[0]
        st.success(f"💰 **Estimated Price:** ${predicted_price:.2f} per night")

# ================================
# DATA VISUALIZATION
# ================================
st.markdown("---")
st.subheader("📊 Data Visualization Dashboard")

col1, col2 = st.columns(2)
with col1:
    st.subheader("🏠 Room Type Distribution")
    room_counts = df['room_type'].value_counts()
    fig1, ax1 = plt.subplots()
    name=['Entire home/apt', 'Private room', 'Shared room']
    ax1.pie(room_counts, labels=name, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')
    st.pyplot(fig1)

with col2:
    st.subheader("📍 Average Price by Neighbourhood Group")
    avg_price = df.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots()
    avg_price.plot(kind='bar', ax=ax2, color="#3498DB", edgecolor="black")
    ax2.set_ylabel("Average Price ($)")
    st.pyplot(fig2)

# ----------------------------
# Data Overview Section
# ----------------------------
st.markdown("### 📊 Data Overview")
st.dataframe(df.head())

# ================================
# PROJECT INTRO SECTION
# ================================
st.markdown("<div id='project-intro'></div>", unsafe_allow_html=True)
st.markdown("---")
st.header("📘 About the Project")

st.markdown("""
### 🎯 **Project Aim**
To develop a **Machine Learning-based predictive model** that can estimate Airbnb listing prices 
based on real-world factors like neighbourhood, room type, minimum nights, and availability.

### 💡 **Introduction**
The **AI-Powered Airbnb Price Prediction Dashboard** uses **Scikit-learn**, **Streamlit**, and **Pandas**
to combine DBMS and ML concepts. It allows users to explore, visualize, and predict prices — 
transforming data into actionable insights.

Built using:
- 🧠 **Machine Learning (Scikit-learn)**
- 🌐 **Streamlit Web Interface**
- 🗂️ **Pandas for Data Handling**
- 🎨 **Matplotlib for Visualizations**
""")

# ================================
# CONTACT SECTION / FOOTER
# ================================
st.markdown("<div id='contact'></div>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div class="footer">
    <p><b>Created by Vinay Kumar - AIML Engineer </b> | © 2025 AI Research & Innovation</p>
    <p>
        <a href="https://github.com/vinaysingh-05" target="_blank">GitHub</a> |
        <a href="https://www.linkedin.com/in/vinay-kumar0805/" target="_blank">LinkedIn</a> |
        <a href="#" target="_blank">Instagram</a>
    </p>
    <p>"Exploring intelligent systems that predict, analyze, and transform real-world data into actionable insights."</p>
</div>
""", unsafe_allow_html=True)