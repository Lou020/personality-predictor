"""Showcase version of the personality app: HUD look, probabilities, model info.

Same pipeline and same MLflow model as app.py, only the interface is richer.
Run from the repo root:  streamlit run ai_app.py
"""
import mlflow
import pandas as pd
import streamlit as st

from personality.config import (
    AGE_MAX, AGE_MIN, CLASS_ORDER, GENDER_OPTIONS, HAND_OPTIONS, ITEM_MAX, ITEM_MIN, QUESTIONS,
)
from personality.plot_style import ACCENT, BACKGROUND, CLASS_COLORS, MUTED, PANEL, TEXT
from personality.tracking import get_best_run_id, load_model

# --- Content ------------------------------------------------------------------
TYPE_INFO = {
    "Moderate": "Balanced. No trait is extreme in either direction.",
    "Resilient": "Emotionally stable and calm under pressure.",
    "Overcontroller": "Anxious and introverted. Tends to hold back.",
    "Undercontroller": "Impulsive and less concerned with rules.",
}
SECTIONS = {
    "01 // EMOTIONAL STABILITY": [c for c in QUESTIONS if c.startswith("N")],
    "02 // SOCIAL ENERGY": [c for c in QUESTIONS if c.startswith("E")],
    "03 // HABITS & EMPATHY": [c for c in QUESTIONS if c[0] in "CA"],
}

# --- Page setup and HUD styling -------------------------------------------------
st.set_page_config(page_title="Personality Predictor", page_icon="◈", layout="wide")
st.markdown(f"""
<style>
/* Canvas: dark with a faint technical grid (graphic realism) */
.stApp {{
    background-color: {BACKGROUND};
    background-image:
        linear-gradient(rgba(92,104,120,0.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(92,104,120,0.07) 1px, transparent 1px);
    background-size: 32px 32px;
}}
html, body, [class*="css"], .stMarkdown, label, p {{ font-family: "DejaVu Sans Mono", monospace !important; }}
h1, h2, h3 {{ color: {ACCENT} !important; letter-spacing: 0.08em; text-transform: uppercase; }}
h3 {{ border-bottom: 1px solid {MUTED}; padding-bottom: 0.3rem; font-size: 1rem !important; }}

/* Hazard stripe + barcode header strip */
.hazard {{ height: 10px; margin: 0.4rem 0 0.8rem 0;
           background: repeating-linear-gradient(-45deg, {ACCENT} 0 12px, {BACKGROUND} 12px 24px); }}
.strip {{ display: flex; justify-content: space-between; align-items: center;
          color: {MUTED}; font-size: 0.72rem; letter-spacing: 0.2em; }}
.barcode {{ width: 140px; height: 22px;
            background: repeating-linear-gradient(90deg, {TEXT} 0 2px, transparent 2px 4px,
                        {TEXT} 4px 5px, transparent 5px 9px, {TEXT} 9px 12px, transparent 12px 14px); }}

/* Panels with corner brackets */
.hud-panel {{ position: relative; background: {PANEL}; border-left: 3px solid {ACCENT};
              padding: 1.1rem 1.3rem; margin-bottom: 1rem; }}
.hud-panel::before, .hud-panel::after {{ content: ""; position: absolute; width: 14px; height: 14px;
              border-color: {MUTED}; border-style: solid; }}
.hud-panel::before {{ top: -1px; right: -1px; border-width: 2px 2px 0 0; }}
.hud-panel::after  {{ bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }}
.hud-label {{ color: {MUTED}; font-size: 0.72rem; letter-spacing: 0.22em; }}
.hud-type {{ font-size: 2.8rem; font-weight: 900; letter-spacing: 0.06em; line-height: 1.1; }}
.tag {{ display: inline-block; padding: 0.1rem 0.5rem; font-size: 0.7rem; letter-spacing: 0.15em;
        color: {BACKGROUND}; font-weight: bold; margin-top: 0.5rem; }}

/* Result panel stays in view while scrolling through the questionnaire */
div[data-testid="stColumn"]:has(.st-key-result) {{ position: sticky; top: 4rem; align-self: flex-start; }}

/* Probability bars */
.bar-row {{ display: flex; align-items: center; margin: 0.45rem 0; font-size: 0.82rem; }}
.bar-name {{ width: 10rem; color: {TEXT}; }}
.bar-track {{ flex: 1; height: 0.8rem; margin: 0 0.6rem;
              background: repeating-linear-gradient(90deg, #1E2633 0 6px, {PANEL} 6px 8px); }}
.bar-fill {{ height: 100%; }}
.bar-val {{ width: 3.5rem; text-align: right; color: {TEXT}; }}
</style>
""", unsafe_allow_html=True)


# --- Model --------------------------------------------------------------------
@st.cache_resource
def get_model_and_info():
    """Load the best model and its MLflow metadata once."""
    run_id = get_best_run_id()
    run = mlflow.get_run(run_id)
    return load_model(run_id), {
        "run_id": run_id,
        "model": run.data.params.get("model", "unknown"),
        "f1": run.data.metrics.get("f1_macro", float("nan")),
    }


model, info = get_model_and_info()


def probability_bars(probabilities: dict) -> str:
    """HTML bar chart: one bar per type in its fixed class color."""
    rows = []
    for name in CLASS_ORDER:
        p = probabilities[name]
        rows.append(
            f'<div class="bar-row"><span class="bar-name">{name}</span>'
            f'<div class="bar-track"><div class="bar-fill" style="width:{p * 100:.1f}%;'
            f'background:{CLASS_COLORS[name]}"></div></div>'
            f'<span class="bar-val">{p * 100:.0f} %</span></div>'
        )
    return "".join(rows)


# --- Header -------------------------------------------------------------------
st.markdown(
    f'<div class="strip"><span>UNIT // PSY-05 &nbsp;&nbsp; BIG FIVE CLASSIFIER</span>'
    f'<div class="barcode"></div></div>',
    unsafe_allow_html=True,
)
st.title("◈ Personality Type Predictor")
st.markdown(
    f'<div class="strip"><span>MODEL: {info["model"].upper()} &nbsp;//&nbsp; '
    f'MACRO F1 {info["f1"]:.3f} &nbsp;//&nbsp; RUN {info["run_id"][:8].upper()}</span>'
    f'<span>STATUS: ONLINE</span></div><div class="hazard"></div>',
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([3, 2], gap="large")

# --- Questionnaire --------------------------------------------------------------
with left:
    with st.form("questionnaire"):
        answers = {}
        for section, codes in SECTIONS.items():
            st.subheader(section)
            for code in codes:
                answers[code] = st.segmented_control(
                    QUESTIONS[code],
                    options=list(range(ITEM_MIN, ITEM_MAX + 1)),
                    default=3,
                    required=True,
                    key=code,
                )

        st.subheader("04 // ABOUT YOU")
        col_age, col_gender, col_hand = st.columns(3)
        answers["age"] = col_age.number_input("Age", min_value=AGE_MIN, max_value=AGE_MAX, value=22)
        answers["gender"] = col_gender.selectbox("Gender", options=GENDER_OPTIONS)
        answers["hand"] = col_hand.selectbox("Writing hand", options=HAND_OPTIONS)

        submitted = st.form_submit_button("▶ ANALYZE", type="primary", use_container_width=True)

# --- Result -------------------------------------------------------------------
with right, st.container(key="result"):
    st.subheader("Result")
    if not submitted:
        st.markdown(
            '<div class="hud-panel"><span class="hud-label">STATUS</span><br>'
            "Awaiting input. Answer the questions and press ANALYZE.</div>",
            unsafe_allow_html=True,
        )
    else:
        input_df = pd.DataFrame([answers])
        probabilities = dict(zip(model.classes_, model.predict_proba(input_df)[0]))
        personality_type = max(probabilities, key=probabilities.get)
        color = CLASS_COLORS[personality_type]

        st.markdown(
            f'<div class="hud-panel" style="border-left-color:{color}">'
            f'<span class="hud-label">PREDICTED TYPE</span><br>'
            f'<span class="hud-type" style="color:{color}">{personality_type.upper()}</span><br>'
            f'{TYPE_INFO[personality_type]}<br>'
            f'<span class="tag" style="background:{color}">CONFIDENCE {probabilities[personality_type] * 100:.0f} %</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="hud-panel"><span class="hud-label">PROBABILITY PER TYPE</span>'
            f"{probability_bars(probabilities)}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            "Note: Undercontroller is the hardest type to detect. The questionnaire has only "
            "one question each for conscientiousness (C4) and agreeableness (A4)."
        )
