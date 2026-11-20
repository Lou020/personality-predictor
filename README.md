# Personality Type Predictor

Predicts a person's personality type (Moderate, Resilient, Overcontroller, Undercontroller) from 19 short questionnaire answers plus age, gender and writing hand. The best model is served in a Streamlit web app.

<!-- WIP: short result summary + recommendation once the model is final -->

## Repository structure

```
personality-predictor/
├── data/              # data.csv + codebook.txt
├── notebooks/         # 01_eda → 04_tuning_mlflow, run in order
├── personality/       # shared code: config, plot style, pipeline
├── models/            # saved best model (created by the notebooks)
├── reports/figures/   # exported charts for the presentation
├── app.py             # Streamlit app
├── requirements.txt
└── pyproject.toml     # makes `personality` importable everywhere
```

## Setup (Windows)

```powershell
git clone https://github.com/<USER>/personality-predictor.git
cd personality-predictor
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

<!-- WIP: macOS/Linux activate command -->

## Run

1. Notebooks in `notebooks/` in numeric order (select the `.venv` kernel)
2. MLflow UI: <!-- WIP: exact command once tracking is set up -->
3. App: `streamlit run app.py`

## Data

Processed version of the IPIP Big Five dataset (Kaggle), see `data/codebook.txt`.
