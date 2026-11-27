import pandas as pd
import streamlit as st

from personality.config import (
    QUESTIONS, ITEM_MIN, ITEM_MAX, AGE_MIN, AGE_MAX, GENDER_OPTIONS, HAND_OPTIONS,
)
from personality.tracking import get_best_run_id, load_model


@st.cache_resource
def get_model():
    """Load the best model from MLflow once and keep it in memory."""
    return load_model(get_best_run_id())


model = get_model()

st.title("Personality Type Predictor")
st.write("Rate each statement from 1 (Disagree) to 5 (Agree).")

answers = {}

#1: 5 Buttons per question, store each answer in `answers`

for question_code, question in QUESTIONS.items():
    answers[question_code] = st.segmented_control(
        question,
        options=list(range(ITEM_MIN, ITEM_MAX + 1)),
        default=3,
        required=True,
    )


#2: age, gender, hand

answers['age'] = st.number_input("Age", min_value=AGE_MIN, max_value=AGE_MAX, value=18)

answers['gender'] = st.selectbox("Gender", options=GENDER_OPTIONS)

answers['hand'] = st.selectbox("Writing hand", options=HAND_OPTIONS)


#3: button -> DataFrame -> predict -> show result

if st.button("Predict"):
    input_df = pd.DataFrame([answers])
    personality_type = model.predict(input_df)[0]
    st.success(f"Your personality type: {personality_type}")