import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import random

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Yoga Guru", layout="wide")

# --- DATA LOADING (Cached for Speed) ---
@st.cache_data
def load_data():
    user_path = r"C:\Users\ragha\OneDrive\Desktop\yoga recommendation\Kaggle Health & Lifestyle Dataset\health_lifestyle_dataset.csv"
    yoga_path = r"C:\Users\ragha\OneDrive\Desktop\yoga recommendation\Kaggle Yoga Poses Recommendation\Yoga Data.xlsx"
    
    df_u = pd.read_csv(user_path)
    df_y = pd.read_excel(yoga_path)
    
    # Preprocess text
    text_cols = ['Benefits', 'Targeted Mental Problems', 'Targeted Physical Problems']
    for col in text_cols:
        df_y[col] = df_y[col].astype(str).replace('nan', '')
    df_y['tags'] = df_y[text_cols].agg(' '.join, axis=1)
    
    return df_u, df_y

df_user, df_yoga = load_data()

# --- NLP ENGINE ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_yoga['tags'])

# --- IMAGE LOCATOR ---
def get_image(asana_name):
    base_path = r"C:\Users\ragha\OneDrive\Desktop\yoga recommendation\Kaggle Yoga Poses Classification"
    # Basic mapping for known mismatches
    mapping = {
    "Padachakrasana": "leg-circles", 
    "Padanguli Naman": "toe-bending",
    "Goolf Chakra": "ankle-rotation",
    "Goolf Ghooman": "ankle-crank",
    "Janufalak Akarshan": "knee-cap-contraction",
    "Bhujangasana": "cobra",
    "Tadasana": "mountain",
    "Adho Mukha Svanasana": "downward-dog"
    }
    folder_key = mapping.get(asana_name, asana_name).lower().replace(" ", "-")
    
    try:
        for folder in os.listdir(base_path):
            if folder_key in folder.lower().replace("_", "-"):
                img_path = os.path.join(base_path, folder)
                imgs = [i for i in os.listdir(img_path) if i.lower().endswith(('.jpg', '.png'))]
                if imgs:
                    return os.path.join(img_path, random.choice(imgs))
    except: return None
    return None

# --- SIDEBAR UI ---
st.sidebar.header("User Health Profile")
user_id = st.sidebar.number_input("Enter User ID", min_value=1, max_value=100000, value=101)

if st.sidebar.button("Generate Recommendations"):
    user = df_user[df_user['id'] == user_id].iloc[0]
    
    # Show User Stats
    st.header(f"Health Analysis for User {user_id}")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("BMI", round(user['bmi'], 1))
    col2.metric("Sleep", f"{user['sleep_hours']} hrs")
    col3.metric("BP", f"{user['systolic_bp']}/{user['diastolic_bp']}")
    col4.metric("Risk Level", str(user['disease_risk']))    # Calculate Recommendations
    query = []
    if user['bmi'] > 25: query.append("weight loss obesity")
    if user['sleep_hours'] < 6: query.append("stress anxiety")
    if user['systolic_bp'] > 130: query.append("hypertension")
    
    user_vec = tfidf.transform([" ".join(query)])
    df_yoga['score'] = cosine_similarity(user_vec, tfidf_matrix).flatten()
    
    # Safety Filter
    results = df_yoga.copy()
    if user['systolic_bp'] > 140:
        results = results[~results['Contraindications'].str.contains("High Blood Pressure", na=False, case=False)]
    
    top_5 = results.sort_values(by='score', ascending=False).head(5)

    # Display Results
    st.divider()
    st.subheader("🧘 Recommended Yoga Routine")
    
    for _, row in top_5.iterrows():
        with st.expander(f"{row['AName']} (Level {row['Level']}) - Match Score: {round(row['score']*100)}%"):
            c1, c2 = st.columns([1, 2])
            img = get_image(row['AName'])
            if img:
                c1.image(img, use_container_width=True)
            else:
                c1.warning("Image placeholder: Folder not found")
            
            c2.write(f"**Benefits:** {row['Benefits']}")
            c2.write(f"**Target Areas:** {row['Target Areas']}")
            if str(row['Contraindications']) != "":
                c2.error(f"**Precautions:** {row['Contraindications']}")

else:
    st.info("Please enter a User ID in the sidebar and click 'Generate' to begin.")
