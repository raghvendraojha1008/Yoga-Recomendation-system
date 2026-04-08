"""
health_chat.py  —  AI Yoga Guru: NLP Health Intake Chat
=========================================================
Drop this file next to app.py. Then in app.py replace the
"New User Profile" sidebar section with:

    from health_chat import render_health_chat
    render_health_chat()

How it works:
  User types symptoms/conditions in plain English.
  Claude extracts structured health data as JSON.
  That JSON is used exactly like the existing current_user dict.
  No sliders. No knowing your exact BMI. Just talk.
"""

import streamlit as st
import json
import re
from feedback_engine import create_user
# ── Anthropic client ──────────────────────────────────────────────────────────
try:
    import anthropic
    _client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY from env
    _HAS_CLAUDE = True
except Exception:
    _HAS_CLAUDE = False


# ── System prompt for extraction ──────────────────────────────────────────────
_EXTRACTION_PROMPT = """You are a health data extraction assistant for a yoga recommendation app.

The user will describe their health situation in plain language.
Your job is to extract structured health data from what they say.

You MUST respond with ONLY a valid JSON object — no explanation, no markdown, no backticks.
Use exactly these keys:

{
  "bmi": <number 10-50, estimate from descriptions like "overweight", "obese", "slim", "normal weight". If unknown use 24.0>,
  "systolic_bp": <number 90-200. "high BP" = 145, "very high" = 160, "low" = 100, "normal" = 120. If unknown use 120>,
  "diastolic_bp": <number 60-120. Estimate from systolic. If unknown use 80>,
  "sleep_hours": <number 2-12. "poor sleep" = 5, "insomnia" = 4, "good sleep" = 7.5. If unknown use 6>,
  "daily_steps": <number 500-20000. "sedentary/desk job" = 2000, "light activity" = 5000, "active" = 10000. If unknown use 4000>,
  "resting_hr": <number 45-110. "high heart rate" = 90, "normal" = 70, "athletic" = 55. If unknown use 72>,
  "cholesterol": <number 120-320. "high cholesterol" = 240, "normal" = 180. If unknown use 185>,
  "smoker": <0 or 1>,
  "alcohol": <0 or 1. "drinks occasionally/regularly" = 1>,
  "family_history": <0 or 1. "family history of diabetes/heart disease/cancer" = 1>,
  "age": <number. Estimate from descriptions like "young", "middle aged", "elderly". If unknown use 35>,
  "gender_enc": <0 for female, 1 for male, 0 if unknown>,
  "disease_risk": <"Low Risk", "Medium Risk", or "High Risk" — your overall assessment>,
  "summary": <one sentence describing the user's health situation in plain English, max 20 words>
}

Rules:
- Always return all 14 keys
- Estimate intelligently from context clues
- Never refuse — always return your best estimate
- If the user says something like "I am healthy" with no issues, use normal values
- disease_risk: Low = healthy/minor issues, Medium = 1-2 chronic issues or poor lifestyle, High = multiple conditions or severe symptoms
"""


_FOLLOWUP_PROMPT = """You are a friendly health intake assistant for a yoga recommendation app.

The user just described their health situation. You extracted this profile:
{profile}

Now do two things in your response:
1. Briefly confirm what you understood (2-3 sentences, warm and conversational tone).
2. Ask ONE specific follow-up question to fill the most important gap in their profile.
   Focus on whatever is most missing: sleep quality, activity level, specific conditions, age, etc.
   Do NOT ask about exact BMI or blood pressure numbers — ask about symptoms and lifestyle instead.

Keep the whole response under 60 words. Friendly, human, not clinical.
"""


# ── Core functions ────────────────────────────────────────────────────────────

def _extract_profile(user_text: str) -> dict:
    """Send user description to Claude, get back structured JSON profile."""
    if not _HAS_CLAUDE:
        return _fallback_profile()

    try:
        msg = _client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=600,
            system=_EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": user_text}]
        )
        raw = msg.content[0].text.strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        profile = json.loads(raw)
        return profile
    except Exception as e:
        st.warning(f"Extraction issue: {e}. Using estimated values.")
        return _fallback_profile()


def _get_followup(profile: dict, original_text: str) -> str:
    """Ask Claude to confirm understanding and ask a follow-up."""
    if not _HAS_CLAUDE:
        return "Got it! Just to confirm — how many hours of sleep do you typically get each night?"

    try:
        summary = profile.get("summary", "your health situation")
        prompt_text = _FOLLOWUP_PROMPT.format(
            profile=json.dumps({
                "summary": profile.get("summary"),
                "disease_risk": profile.get("disease_risk"),
                "bmi": profile.get("bmi"),
                "sleep_hours": profile.get("sleep_hours"),
                "daily_steps": profile.get("daily_steps"),
            }, indent=2)
        )
        msg = _client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt_text}]
        )
        return msg.content[0].text.strip()
    except Exception:
        return "Thanks for sharing! How many hours do you sleep on average per night?"


def _fallback_profile() -> dict:
    """Default profile when API is unavailable."""
    return {
        "bmi": 24.0, "systolic_bp": 120, "diastolic_bp": 80,
        "sleep_hours": 6.0, "daily_steps": 4000, "resting_hr": 72,
        "cholesterol": 185, "smoker": 0, "alcohol": 0,
        "family_history": 0, "age": 35, "gender_enc": 0,
        "disease_risk": "Medium Risk",
        "summary": "General health profile — update by describing your situation."
    }


def _risk_color(risk: str) -> str:
    return {"Low Risk": "green", "Medium Risk": "orange", "High Risk": "red"}.get(risk, "gray")


# ── Main Streamlit component ──────────────────────────────────────────────────

def render_health_chat():
    """
    Call this from app.py inside the 'New User Profile' sidebar section.
    Manages its own session state. Sets st.session_state.current_user
    when a profile is confirmed.
    """

    # Init session state
    if "hc_messages" not in st.session_state:
        st.session_state.hc_messages = []
    if "hc_profile" not in st.session_state:
        st.session_state.hc_profile = None
    if "hc_confirmed" not in st.session_state:
        st.session_state.hc_confirmed = False
    if "hc_turn" not in st.session_state:
        st.session_state.hc_turn = 0  # 0=initial, 1=followup done, 2=confirmed

    st.markdown("### 💬 Tell me about your health")
    st.caption("No need to know exact numbers — just describe how you feel, your lifestyle, any conditions.")

    # Show chat history
    for msg in st.session_state.hc_messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # --- TURN 0: Initial intake ---
    if st.session_state.hc_turn == 0:
        placeholder_text = (
            "e.g. I'm 45, slightly overweight, desk job, get maybe 5 hours sleep, "
            "doctor says my BP is a bit high, no major conditions..."
        )
        user_input = st.chat_input(placeholder_text)

        if user_input:
            # Show user message
            st.session_state.hc_messages.append({"role": "user", "content": user_input})

            with st.spinner("Understanding your health profile..."):
                profile = _extract_profile(user_input)
                followup = _get_followup(profile, user_input)

            st.session_state.hc_profile = profile
            st.session_state.hc_messages.append({"role": "assistant", "content": followup})
            st.session_state.hc_turn = 1
            st.rerun()

    # --- TURN 1: Follow-up response ---
    elif st.session_state.hc_turn == 1:
        user_input = st.chat_input("Your answer...")

        if user_input:
            st.session_state.hc_messages.append({"role": "user", "content": user_input})

            # Re-extract with combined context for better accuracy
            combined = "\n".join([
                m["content"] for m in st.session_state.hc_messages
                if m["role"] == "user"
            ])

            with st.spinner("Updating your profile..."):
                refined_profile = _extract_profile(combined)

            st.session_state.hc_profile = refined_profile
            risk = refined_profile.get("disease_risk", "Medium Risk")
            summary = refined_profile.get("summary", "")
            color = _risk_color(risk)

            confirm_msg = (
                f"✅ Profile ready!\n\n"
                f"**Health summary:** {summary}\n\n"
                f"**Risk level:** :{color}[{risk}]\n\n"
                f"Click **Generate My Routine** below to see your personalised yoga plan."
            )
            st.session_state.hc_messages.append({"role": "assistant", "content": confirm_msg})
            st.session_state.hc_turn = 2
            st.rerun()

    # --- TURN 2: Show profile + confirm button ---
    elif st.session_state.hc_turn == 2:
        profile = st.session_state.hc_profile
        if profile:
            st.divider()

            # Show key extracted values so user can verify
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Est. BMI", profile.get("bmi", "—"))
                st.metric("Sleep", f"{profile.get('sleep_hours', '—')}h")
                st.metric("Activity", f"{profile.get('daily_steps', '—'):,} steps/day")
            with col2:
                st.metric("Systolic BP", profile.get("systolic_bp", "—"))
                st.metric("Age (est.)", profile.get("age", "—"))
                risk = profile.get("disease_risk", "Medium Risk")
                color = _risk_color(risk)
                st.markdown(f"**Risk:** :{color}[{risk}]")

            st.caption("These are estimates from your description. The system uses them to find the right poses.")

            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("🧘 Generate My Routine", type="primary", use_container_width=True):
                    # Set the current_user in session state — app.py picks this up
                    

                    user_id = create_user(profile)

                    profile["user_id"] = user_id

                    st.session_state.current_user = profile
                    st.session_state.top_recs = None
                    st.session_state.hc_confirmed = True
                    st.rerun()

            with col_btn2:
                if st.button("🔄 Start Over", use_container_width=True):
                    for key in ["hc_messages", "hc_profile", "hc_confirmed", "hc_turn"]:
                        del st.session_state[key]
                    st.rerun()
