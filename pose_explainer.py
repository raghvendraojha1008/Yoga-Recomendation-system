"""
pose_explainer.py  —  AI Yoga Guru: LLM Pose Explainer
========================================================
Calls Claude API to generate a personalised explanation of WHY
a specific yoga pose is recommended for a specific user's health profile.

Usage in app.py:
    from pose_explainer import get_pose_explanation, render_explanation_button

The key function is get_pose_explanation(user, pose_row) -> str
It returns 3-4 sentences explaining the match in plain English.

Cached per (user_risk + pose_name) so the same explanation is not
regenerated on every Streamlit rerun.
"""

import streamlit as st
import re

# ── Anthropic client setup ────────────────────────────────────────────────────
try:
    import anthropic
    _client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env
    _HAS_CLAUDE = True
except Exception:
    _HAS_CLAUDE = False


# ── System prompt ─────────────────────────────────────────────────────────────
_SYSTEM = """You are an expert yoga therapist and health coach.
Your job is to explain in plain, warm language why a specific yoga pose
is a good match for a specific person's health situation.

Rules:
- Write exactly 3 sentences. No more, no less.
- First sentence: connect the pose directly to the user's main health concern.
- Second sentence: explain one specific physiological or psychological benefit.
- Third sentence: mention any precaution or tip for this user given their profile.
- Do NOT use generic phrases like "this pose is great for everyone".
- Be specific to the numbers and conditions given.
- Tone: warm, knowledgeable, like a real yoga instructor talking to a patient.
- Do not start with "This pose" — vary the opening.
"""


# ── Cache: avoid re-calling API for same pose+risk combo ─────────────────────
# Key = (disease_risk, pose_name)  →  explanation string
_explanation_cache: dict[tuple, str] = {}


def _build_user_summary(user: dict) -> str:
    """Convert user dict to a readable health summary for the prompt."""
    parts = []

    bmi = user.get('bmi')
    if bmi:
        if bmi < 18.5:
            parts.append(f"BMI {bmi:.1f} (underweight)")
        elif bmi < 25:
            parts.append(f"BMI {bmi:.1f} (healthy weight)")
        elif bmi < 30:
            parts.append(f"BMI {bmi:.1f} (overweight)")
        else:
            parts.append(f"BMI {bmi:.1f} (obese range)")

    sys_bp = user.get('systolic_bp')
    if sys_bp:
        if sys_bp >= 140:
            parts.append(f"high blood pressure ({sys_bp} systolic)")
        elif sys_bp >= 130:
            parts.append(f"borderline high BP ({sys_bp} systolic)")
        else:
            parts.append(f"normal BP ({sys_bp} systolic)")

    sleep = user.get('sleep_hours')
    if sleep:
        if sleep < 5:
            parts.append(f"severe sleep deprivation ({sleep:.0f}h/night)")
        elif sleep < 6.5:
            parts.append(f"poor sleep ({sleep:.0f}h/night)")
        else:
            parts.append(f"adequate sleep ({sleep:.0f}h/night)")

    steps = user.get('daily_steps')
    if steps:
        if steps < 3000:
            parts.append("very sedentary lifestyle")
        elif steps < 6000:
            parts.append("low activity level")
        else:
            parts.append("moderately active")

    smoker = user.get('smoker', 0)
    if smoker:
        parts.append("smoker")

    risk = user.get('disease_risk', 'Medium Risk')
    parts.append(f"overall health risk: {risk}")

    summary = user.get('summary', '')  # from NLP chat intake if available

    if summary:
        return f"{summary}. Additional details: {', '.join(parts)}."
    return ', '.join(parts) + '.'


def get_pose_explanation(user: dict, pose_name: str,
                         benefits: str, target_areas: str,
                         contraindications: str) -> str:
    """
    Generate a personalised explanation of why pose_name suits this user.

    Parameters
    ----------
    user             : the current_user dict from session state
    pose_name        : e.g. "Shavasana"
    benefits         : from the yoga dataset row
    target_areas     : from the yoga dataset row
    contraindications: from the yoga dataset row

    Returns
    -------
    str : 3-sentence personalised explanation, or a safe fallback string.
    """
    risk  = user.get('disease_risk', 'Medium Risk')
    cache_key = (risk, pose_name)

    # Return cached result if available
    if cache_key in _explanation_cache:
        return _explanation_cache[cache_key]

    # Fallback if Claude not available
    if not _HAS_CLAUDE:
        fallback = (
            f"{pose_name} is recommended based on your health profile. "
            f"It targets {target_areas or 'the full body'} and provides "
            f"benefits including {benefits[:120] if benefits else 'overall wellness'}. "
            f"{'Note: ' + contraindications[:80] if contraindications and contraindications.lower() not in ['none','nan',''] else 'No major precautions for this pose.'}"
        )
        _explanation_cache[cache_key] = fallback
        return fallback

    # Build the prompt
    user_summary = _build_user_summary(user)

    # Clean contraindications for the prompt
    contra_clean = contraindications.strip() if contraindications and \
                   contraindications.lower() not in ['none', 'nan', ''] else 'none known'

    prompt = f"""User health profile: {user_summary}

Recommended pose: {pose_name}
Pose benefits: {benefits[:300] if benefits else 'general wellness'}
Target areas: {target_areas or 'full body'}
Contraindications: {contra_clean}

Write exactly 3 sentences explaining why {pose_name} is specifically suited
to this user's health situation."""

    try:
        msg = _client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=200,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}]
        )
        explanation = msg.content[0].text.strip()
        # Ensure no stray markdown
        explanation = re.sub(r'\*\*|__', '', explanation)
        _explanation_cache[cache_key] = explanation
        return explanation

    except Exception as e:
        fallback = (
            f"{pose_name} aligns well with your health profile. "
            f"It supports {target_areas or 'full body'} health "
            f"through {benefits[:100] if benefits else 'mindful movement'}."
        )
        _explanation_cache[cache_key] = fallback
        return fallback


def render_explanation_button(user: dict, pose_name: str,
                               benefits: str, target_areas: str,
                               contraindications: str,
                               container) -> None:
    """
    Renders a 'Why this pose for me?' button inside the given Streamlit
    container (e.g. a column). On click, fetches and displays the explanation.

    Uses session state key  'expl_{pose_name}'  to persist the result
    across reruns without re-calling the API.

    Parameters
    ----------
    container : a streamlit column or container to write into
    """
    safe_key = re.sub(r'\W+', '_', pose_name)   # make key safe for st keys
    state_key = f"expl_{safe_key}"

    if state_key not in st.session_state:
        st.session_state[state_key] = None

    with container:
        if st.session_state[state_key] is None:
            if st.button(
                "🤖 Why this pose for me?",
                key=f"btn_{safe_key}",
                help="Ask Claude AI why this specific pose matches your health profile"
            ):
                with st.spinner("Claude is analysing your profile..."):
                    explanation = get_pose_explanation(
                        user, pose_name, benefits, target_areas, contraindications
                    )
                st.session_state[state_key] = explanation
                st.rerun()
        else:
            # Show the explanation in a styled box
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, #f0f7f0 0%, #e8f4e8 100%);
                    border-left: 4px solid #2e7d32;
                    border-radius: 0 8px 8px 0;
                    padding: 12px 16px;
                    margin: 8px 0;
                    font-size: 0.92em;
                    line-height: 1.6;
                    color: #1b5e20;
                ">
                🧠 <strong>Why this pose suits you:</strong><br><br>
                {st.session_state[state_key]}
                </div>
                """,
                unsafe_allow_html=True
            )
            # Small button to regenerate if user wants a fresh explanation
            if st.button("↻ Regenerate", key=f"regen_{safe_key}",
                         help="Get a new explanation"):
                # Remove from both caches
                st.session_state[state_key] = None
                risk  = user.get('disease_risk', 'Medium Risk')
                cache_key = (risk, pose_name)
                if cache_key in _explanation_cache:
                    del _explanation_cache[cache_key]
                st.rerun()
