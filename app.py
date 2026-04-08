import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os, random, json, difflib, joblib, time
import plotly.graph_objects as go
from fpdf import FPDF
import numpy as np
import cv2
from yoga_utils import YogaCoach
from live_pose import LivePoseDetector
from streamlit_webrtc import webrtc_streamer
# ── NEW: import the health chat module ───────────────────────────────────────
from health_chat import render_health_chat
from pose_explainer import render_explanation_button
from feedback_engine import (init_db, get_reranked_recs,
                             render_feedback_section,
                             render_learning_sidebar)

# --- APP CONFIG ---
st.set_page_config(page_title="AI Yoga Guru", layout="wide", page_icon="🧘")
init_db()   # create yoga_feedback.db if not exists

@st.cache_resource
def init_coach():
    return YogaCoach()

coach = init_coach()

# Cache live pose detector
@st.cache_resource
def load_pose_model():
    return LivePoseDetector()

pose_detector = load_pose_model()

if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'top_recs' not in st.session_state:
    st.session_state.top_recs = None

# --- ASSET LOADING ---
@st.cache_resource
def load_assets():
    try:
        with open('metadata.json', 'r') as f:
            metadata = json.load(f)
    except:
        metadata = {"pose_image_mapping": {}, "target_area_mapping": {}}
    model_path = 'disease_model.pkl'
    model = joblib.load(model_path) if os.path.exists(model_path) else None
    return metadata, model

metadata, disease_model = load_assets()
pose_map = metadata.get('pose_image_mapping', {})
target_map = metadata.get('target_area_mapping', {})

# --- DATA LOADING ---
@st.cache_data
def load_data():
    # Update these paths to match your local setup
    u_path = "C:\\Users\\ragha\\OneDrive\\Desktop\\yoga recommendation\\Kaggle Health & Lifestyle Dataset\\health_lifestyle_dataset.csv"
    y_path = "C:\\Users\\ragha\\OneDrive\\Desktop\\yoga recommendation\\Kaggle Yoga Poses Recommendation\\Yoga Data.xlsx"
    df_u = pd.read_csv(u_path)
    df_y = pd.read_excel(y_path)
    text_cols = ['Benefits', 'Targeted Mental Problems', 'Targeted Physical Problems']
    for col in text_cols:
        df_y[col] = df_y[col].astype(str).replace('nan', '')
    df_y['tags'] = df_y[text_cols].agg(' '.join, axis=1)
    return df_u, df_y

df_user, df_yoga = load_data()

# --- NLP ENGINE ---
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df_yoga['tags'])

# --- HELPERS ---
def get_image(asana_name):
    base_path = r"C:\Users\ragha\OneDrive\Desktop\yoga recommendation\Kaggle Yoga Pose Classification"
    search_term = pose_map.get(asana_name, asana_name)
    try:
        all_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
        matches = difflib.get_close_matches(search_term, all_folders, n=1, cutoff=0.3)
        if matches:
            folder = matches[0]
            internal = folder.split('-202')[0]
            img_dir = os.path.join(base_path, folder, internal)
            imgs = [i for i in os.listdir(img_dir) if i.lower().endswith(('.jpg', '.png', '.jpeg'))]
            return os.path.join(img_dir, random.choice(imgs)) if imgs else None
    except:
        return None

def generate_pdf(user, recs):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "AI Yoga Guru: Your Routine", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    summary = user.get('summary', '')
    if summary:
        pdf.multi_cell(0, 8, f"Health summary: {summary}")
        pdf.ln(4)
    pdf.cell(200, 10, f"BMI: {user.get('bmi','?')} | Risk: {user.get('disease_risk','?')}", ln=True)
    pdf.ln(10)
    for _, row in recs.iterrows():
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(200, 10, f"Pose: {row['AName']}", ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 10, f"Benefits: {row['Benefits']}")
        # Include cached AI explanation if available (no extra API call)
        from pose_explainer import _explanation_cache
        risk_key = user.get('disease_risk', 'Medium Risk')
        expl = _explanation_cache.get((risk_key, str(row['AName'])), '')
        if expl:
            pdf.set_font("Arial", 'I', 10)
            pdf.multi_cell(0, 8, f"Why this suits you: {expl}")
        pdf.ln(5)
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR ---
st.sidebar.title("🧘 AI Yoga Guru")
mode = st.sidebar.radio("How do you want to start?", ["Existing User", "Chat with AI"])
lvl  = st.sidebar.selectbox("Difficulty Level", ["All Levels", 1, 2, 3])

# ── EXISTING USER (unchanged) ─────────────────────────────────────────────────
if mode == "Existing User":
    u_id = st.sidebar.number_input("Enter User ID", min_value=1, value=101)
    if st.sidebar.button("Fetch Analysis"):
        st.session_state.current_user = df_user[df_user['id'] == u_id].iloc[0].to_dict()
        st.session_state.top_recs = None

# ── NEW: CHAT-BASED INTAKE ────────────────────────────────────────────────────
else:
    # render_health_chat() lives in the MAIN area (not sidebar) because
    # it uses st.chat_input which doesn't work inside st.sidebar
    pass  # actual render happens below in main area

# --- MAIN AREA ---

# Show chat intake UI when in chat mode and no user yet confirmed
if mode == "Chat with AI" and not st.session_state.current_user:
    render_health_chat()

# Show dashboard once we have a user (either mode)
elif st.session_state.current_user:
    u = st.session_state.current_user

    # --- HEALTH DASHBOARD ---
    st.header("🏥 Health Dashboard")

    # Show the NLP-derived summary if available (new feature)
    if u.get('summary'):
        st.info(f"**AI Health Summary:** {u['summary']}")

    # BMI Gauge
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=u.get('bmi', 24),
        title={'text': "BMI Category"},
        gauge={'axis': {'range': [10, 50]}, 'steps': [
            {'range': [10, 18.5], 'color': "lightblue"},
            {'range': [18.5, 25], 'color': "green"},
            {'range': [25, 30],  'color': "yellow"},
            {'range': [30, 50],  'color': "red"},
        ]}
    ))
    st.plotly_chart(fig, use_container_width=True)

    # AI Disease Prediction (for existing users loaded by ID)
    if disease_model and u.get('disease_risk') == "Analyzing...":
        try:
            u['disease_risk'] = disease_model.predict([[
                u.get('bmi', 24), u.get('systolic_bp', 120),
                u.get('diastolic_bp', 80), u.get('sleep_hours', 7)
            ]])[0]
        except Exception:
            u['disease_risk'] = 'Medium Risk'

    # Risk badge
    risk = u.get('disease_risk', 'Medium Risk')
    color_map = {"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"}
    badge_color = color_map.get(risk, "gray")
    st.markdown(f"**Health Risk Level:** :{badge_color}[{risk}]")

    # ── Adaptive learning progress in sidebar ──────────────
    render_learning_sidebar(risk)

    # --- RECOMMENDATION ENGINE ---
    if st.session_state.top_recs is None:
        bmi_val = u.get('bmi', 24)
        bp_val  = u.get('systolic_bp', 120)
        slp_val = u.get('sleep_hours', 7)

        # Map risk level to therapeutic keywords for better NLP matching
        risk_keywords = {
            "Low Risk":    "flexibility strength balance",
            "Medium Risk": "stress relief gentle stretching relaxation",
            "High Risk":   "relaxation breathing restorative calm",
        }
        extra = risk_keywords.get(risk, "")
        q = f"bmi {bmi_val} blood pressure {bp_val} sleep {slp_val} {extra}"

        vec = tfidf.transform([q])
        df_yoga['score'] = cosine_similarity(vec, tfidf_matrix).flatten()
        res = df_yoga.copy()
        if lvl != "All Levels":
            res = res[res['Level'] == lvl]

        # ── ADAPTIVE: re-rank using stored user ratings ────────────────
        # On first visit (no ratings) this is identical to pure cosine.
        # After the user rates sessions it gradually personalises.
        st.session_state.top_recs = get_reranked_recs(res, risk, top_n=5)

    st.subheader("🧘 Personalized Recommendations")
    for _, row in st.session_state.top_recs.iterrows():
        pose_name   = str(row['AName'])
        benefits    = str(row.get('Benefits', ''))
        target_str  = target_map.get(str(row.get('Target Areas', '')), 'Full Body')
        contra_str  = str(row.get('Contraindications', 'none'))

        with st.expander(f"{pose_name} (Level {row['Level']})"):
            c1, c2 = st.columns([1, 2])

            img_p = get_image(pose_name)
            if img_p:
                c1.image(img_p, use_container_width=True)

            # ── existing info ──────────────────────────────────────────
            c2.write(f"**Target:** {target_str}")
            c2.write(f"**Benefits:** {benefits}")
            if contra_str.lower() not in ["none", "nan", ""]:
                c2.error(f"**Safety:** {contra_str}")

            # ── NEW: LLM personalised explanation ─────────────────────
            render_explanation_button(
                user        = u,
                pose_name   = pose_name,
                benefits    = benefits,
                target_areas= target_str,
                contraindications = contra_str,
                container   = c2
            )

    # --- PDF DOWNLOAD ---
    st.divider()
    pdf_data = generate_pdf(u, st.session_state.top_recs)
    st.download_button("📥 Download Plan as PDF", data=pdf_data, file_name="yoga_plan.pdf")

    # --- FEEDBACK / RATING SECTION ---
    render_feedback_section(st.session_state.top_recs, risk)

    # --- LIVE POSE CORRECTION ---
# --- LIVE POSE CORRECTION ---
    st.divider()
    st.subheader("📸 AI Live Pose Correction")

    st.info("Stand 2–3 meters away so your full body is visible to the camera.")

    pose_list = []

    if st.session_state.top_recs is not None:
        pose_list = st.session_state.top_recs['AName'].astype(str).tolist()

    pose_name = st.selectbox(
        "Select Pose to Verify",
        pose_list
    )

    
    webrtc_streamer(
        key="yoga-live",
        video_processor_factory=lambda: pose_detector,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 30}
            },
            "audio": False
        },
        async_processing=True
    )

    # --- SWITCH PROFILE BUTTON ---
    st.divider()
    if st.button("← Start Over / Change Profile"):
        # Clear all session state including explanation cache keys
        keys_to_clear = ["current_user", "top_recs",
                         "hc_messages", "hc_profile", "hc_confirmed", "hc_turn"]
        # Also clear any cached pose explanations
        expl_keys = [k for k in st.session_state if k.startswith("expl_")]
        for key in keys_to_clear + expl_keys:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

else:
    st.info("👈 Choose 'Chat with AI' for a personalised experience, or enter a User ID.")
