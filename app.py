from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="HeartCare AI | Heart Health Assessment",
    page_icon="💙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --navy: #142b4a;
        --teal: #0d9488;
        --blue: #3983bd;
        --muted: #64748b;
        --canvas: #f4f7fa;
    }
    .stApp {
        background: var(--canvas);
        color: #26364a;
    }
    [data-testid="stHeader"] {
        background: rgba(244, 247, 250, 0.92);
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5ebf1;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #52647a;
    }
    h1, h2, h3 {
        color: var(--navy) !important;
        letter-spacing: -0.025em;
    }
    .brand-mark {
        color: var(--teal);
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        margin: 0 0 0.15rem;
    }
    .sidebar-credit {
        color: var(--muted);
        font-size: 0.84rem;
        margin: 0 0 1.5rem;
    }
    .sidebar-credit strong {
        color: var(--navy);
        font-weight: 700;
    }
    .sidebar-credit strong span {
        color: var(--teal);
    }
    .hero {
        background: linear-gradient(115deg, #ffffff 0%, #eff8f8 100%);
        border: 1px solid #e4eeee;
        border-radius: 20px;
        padding: 1.8rem 2rem;
        margin: 0.4rem 0 1.5rem;
        box-shadow: 0 8px 26px rgba(20, 43, 74, 0.045);
    }
    .eyebrow {
        color: var(--teal);
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.55rem;
    }
    .hero h1 {
        font-size: clamp(1.8rem, 4vw, 2.6rem);
        margin: 0;
    }
    .hero-copy {
        color: #53667a;
        font-size: 1rem;
        margin: 0.5rem 0 0;
    }
    .hero-credit {
        color: var(--muted);
        font-size: 0.78rem;
        font-weight: 600;
        margin: 0.9rem 0 0;
    }
    .hero-credit strong, .footer-credit strong {
        color: var(--navy);
        font-weight: 700;
    }
    .hero-credit strong span, .footer-credit strong span {
        color: var(--teal);
    }
    .section-caption {
        color: var(--muted);
        margin-top: -0.65rem;
        margin-bottom: 1rem;
        font-size: 0.94rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border: 1px solid #e7edf2;
        border-radius: 16px;
        box-shadow: 0 5px 20px rgba(20, 43, 74, 0.045);
    }
    [data-testid="stWidgetLabel"] p {
        color: #30445b;
        font-weight: 600;
    }
    [data-testid="stNumberInput"] input,
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stSlider"] {
        border-radius: 10px;
    }
    div.stButton > button[kind="primary"] {
        min-height: 3.1rem;
        border: 0;
        border-radius: 11px;
        background: linear-gradient(100deg, #0d9488, #087f8c);
        color: #ffffff;
        font-weight: 700;
        box-shadow: 0 6px 14px rgba(13, 148, 136, 0.2);
        transition: transform 140ms ease, box-shadow 140ms ease;
    }
    div.stButton > button[kind="primary"]:hover {
        background: linear-gradient(100deg, #0b8279, #076f7b);
        transform: translateY(-1px);
        box-shadow: 0 8px 17px rgba(13, 148, 136, 0.25);
    }
    .result-title {
        color: var(--navy);
        font-size: 1.25rem;
        font-weight: 750;
        margin-bottom: 0.35rem;
    }
    .result-copy {
        color: #52647a;
        margin: 0;
    }
    .disclaimer, .footer {
        color: #64748b;
        font-size: 0.82rem;
        line-height: 1.55;
    }
    .footer {
        border-top: 1px solid #e2e8f0;
        margin-top: 2.5rem;
        padding-top: 1rem;
        text-align: center;
    }
    .footer-credit {
        color: var(--muted);
        font-size: 0.8rem;
        margin: 0 0 0.25rem;
    }
    @media (max-width: 640px) {
        .hero {
            padding: 1.35rem 1.15rem;
            border-radius: 15px;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            border-radius: 13px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    """Load the existing trained model and preprocessing artifacts once."""
    model = joblib.load(APP_DIR / "knn_heart_model.pkl")
    scaler = joblib.load(APP_DIR / "heart_scaler.pkl")
    expected_columns = joblib.load(APP_DIR / "heart_columns.pkl")
    return model, scaler, expected_columns


with st.sidebar:
    st.markdown('<div class="brand-mark">💙 HeartCare AI</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sidebar-credit">Developed by '
        '<strong>Mohammed <span>Khan</span></strong></p>',
        unsafe_allow_html=True,
    )
    st.markdown("### Health tools")
    page = st.radio(
        "Navigation",
        ["Heart assessment", "About this tool"],
        label_visibility="collapsed",
    )
    st.divider()
    st.caption("A machine-learning educational experience.")

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">💙 HeartCare AI</div>
      <h1>Heart Health Assessment</h1>
      <p class="hero-copy">Machine Learning-Based Heart Disease Prediction</p>
      <p class="hero-credit">Designed &amp; Developed by
      <strong>Mohammed <span>Khan</span></strong></p>
    </section>
    """,
    unsafe_allow_html=True,
)

if page == "About this tool":
    st.subheader("A thoughtful starting point for heart-health learning")
    st.write(
        "HeartCare AI applies a previously trained machine-learning model to "
        "the information entered in the assessment. It is designed for "
        "educational exploration and is not a substitute for clinical care."
    )
    st.info(
        "The model returns a class label that this app presents as lower or "
        "elevated model-indicated risk. It does not provide a diagnosis, "
        "probability, or confidence estimate."
    )
else:
    st.subheader("Patient Information")
    st.markdown(
        '<p class="section-caption">Enter the patient details used by the saved model.</p>',
        unsafe_allow_html=True,
    )

    with st.container(border=True):
        patient_col, clinical_col = st.columns(2, gap="large")
        with patient_col:
            st.markdown("#### Basic details")
            age = st.slider("Age", 18, 100, 40, help="Patient age in years.")
            sex = st.selectbox(
                "Sex",
                ["M", "F"],
                format_func=lambda value: "Male" if value == "M" else "Female",
                help="Sex value used by the model's original encoding.",
            )
        with clinical_col:
            st.markdown("#### Clinical measurements")
            resting_bp = st.number_input(
                "Resting Blood Pressure (mm Hg)",
                min_value=80,
                max_value=200,
                value=120,
                help="Resting systolic blood pressure.",
            )
            cholesterol = st.number_input(
                "Cholesterol (mg/dL)",
                min_value=100,
                max_value=600,
                value=200,
                help="Serum cholesterol measurement.",
            )
            fasting_bs = st.selectbox(
                "Fasting Blood Sugar > 120 mg/dL",
                [0, 1],
                format_func=lambda value: "Yes" if value == 1 else "No",
                help="Select Yes if fasting blood sugar is above 120 mg/dL.",
            )
            max_hr = st.slider(
                "Maximum Heart Rate (bpm)",
                60,
                220,
                150,
                help="Maximum heart rate recorded during exercise.",
            )
            oldpeak = st.slider(
                "Oldpeak (ST depression)",
                0.0,
                6.0,
                1.0,
                step=0.1,
                help="ST depression induced by exercise relative to rest.",
            )

    st.subheader("Heart Assessment")
    st.markdown(
        '<p class="section-caption">Clinical observations and exercise response.</p>',
        unsafe_allow_html=True,
    )
    with st.container(border=True):
        assessment_left, assessment_right = st.columns(2, gap="large")
        with assessment_left:
            chest_pain = st.selectbox(
                "Chest Pain Type",
                ["ATA", "NAP", "TA", "ASY"],
                format_func=lambda value: {
                    "ATA": "Atypical angina",
                    "NAP": "Non-anginal pain",
                    "TA": "Typical angina",
                    "ASY": "Asymptomatic",
                }[value],
                help="Select the chest pain category recorded for the patient.",
            )
            resting_ecg = st.selectbox(
                "Resting ECG",
                ["Normal", "ST", "LVH"],
                format_func=lambda value: {
                    "Normal": "Normal",
                    "ST": "ST-T wave abnormality",
                    "LVH": "Left ventricular hypertrophy",
                }[value],
                help="Resting electrocardiogram result.",
            )
        with assessment_right:
            exercise_angina = st.selectbox(
                "Exercise-Induced Angina",
                ["Y", "N"],
                format_func=lambda value: "Yes" if value == "Y" else "No",
                help="Whether angina was induced by exercise.",
            )
            st_slope = st.selectbox(
                "ST Slope",
                ["Up", "Flat", "Down"],
                help="Slope of the peak exercise ST segment.",
            )

    predict_clicked = st.button("💙  Predict", type="primary", use_container_width=True)

    if predict_clicked:
        try:
            model, scaler, expected_columns = load_artifacts()

            # Keep the original one-hot feature names and preprocessing unchanged.
            raw_input = {
                "Age": age,
                "RestingBP": resting_bp,
                "Cholesterol": cholesterol,
                "FastingBS": fasting_bs,
                "MaxHR": max_hr,
                "Oldpeak": oldpeak,
                "Sex_" + sex: 1,
                "ChestPainType_" + chest_pain: 1,
                "RestingECG_" + resting_ecg: 1,
                "ExerciseAngina_" + exercise_angina: 1,
                "ST_Slope_" + st_slope: 1,
            }
            input_df = pd.DataFrame([raw_input])

            for column in expected_columns:
                if column not in input_df.columns:
                    input_df[column] = 0

            input_df = input_df[expected_columns]
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0]

            st.subheader("Assessment Result")
            with st.container(border=True):
                if prediction == 1:
                    st.warning(
                        "### ⚠️ Elevated model-indicated risk\n\n"
                        "The model output is class 1, which the existing app "
                        "labels as high risk of heart disease."
                    )
                elif prediction == 0:
                    st.success(
                        "### ✅ Lower model-indicated risk\n\n"
                        "The model output is class 0, which the existing app "
                        "labels as low risk of heart disease."
                    )
                else:
                    st.info(f"Model output class: {prediction}")
                st.markdown(
                    '<p class="disclaimer"><strong>Important:</strong> This '
                    'educational machine-learning result is not a medical '
                    'diagnosis or a substitute for advice from a qualified '
                    'healthcare professional. Seek professional care for '
                    'health concerns.</p>',
                    unsafe_allow_html=True,
                )
        except FileNotFoundError as error:
            st.error(
                f"Could not load a required model file: {error.filename}. "
                "Place all three .pkl files in the same folder as app.py."
            )
        except Exception as error:
            st.error(
                f"The assessment could not be completed ({type(error).__name__}: "
                f"{error}). Please check the model files and entered values, "
                "then try again."
            )

st.markdown(
    """
    <div class="footer">
      <p class="footer-credit">© 2026 <strong>Mohammed <span>Khan</span></strong> | HeartCare AI</p>
      HeartCare AI · Educational machine-learning tool, not medical advice.
      Always consult a qualified healthcare professional about your health.
    </div>
    """,
    unsafe_allow_html=True,
)
