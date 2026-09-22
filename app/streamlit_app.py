"""
Titanic ML — Interactive Streamlit Dashboard
==============================================
Interactive web interface for passenger survival prediction and model insights.

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import pandas as pd
import streamlit as st

from src.config import Config
from src.data_loader import load_data
from src.prediction import load_model, predict_single_passenger

# Page configuration
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_config() -> Config:
    """Initialize and cache configuration."""
    return Config()


@st.cache_resource
def get_model_artifacts(_config: Config):
    """Load and cache model artifacts."""
    try:
        return load_model(config=_config)
    except FileNotFoundError:
        return None


@st.cache_data
def get_dataset(_config: Config):
    """Load training and test datasets."""
    try:
        return load_data(_config)
    except Exception:
        return None, None


def main() -> None:
    config = get_config()
    artifacts = get_model_artifacts(config)
    train_df, test_df = get_dataset(config)

    st.title("🚢 Titanic ML — Passenger Survival Predictor")
    st.markdown(
        "A production-grade machine learning solution powered by a "
        "**Stacking Ensemble** of 7 diverse models with advanced feature engineering."
    )

    # Sidebar: Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["Survival Simulator", "Model Performance", "Exploratory Data Analysis"],
    )

    if page == "Survival Simulator":
        render_simulator(config, artifacts)
    elif page == "Model Performance":
        render_performance(artifacts)
    elif page == "Exploratory Data Analysis":
        render_eda(train_df)


def render_simulator(config: Config, artifacts: dict | None) -> None:
    """Render the interactive passenger survival prediction form."""
    st.header("🔮 Passenger Survival Simulator")
    st.write(
        "Configure passenger attributes to predict their survival probability using the trained ensemble model."
    )

    if artifacts is None:
        st.warning(
            "⚠️ No trained model found in `models/titanic_model.joblib`. "
            "Please run `python train.py` first to train and save the model."
        )
        return

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Demographics")
        sex = st.selectbox("Sex", ["female", "male"])
        age = st.slider("Age", min_value=0.5, max_value=80.0, value=28.0, step=0.5)
        title = st.selectbox(
            "Honorific Title",
            ["Mr", "Miss", "Mrs", "Master", "Rare (Dr/Rev/Col/etc.)"],
        )

    with col2:
        st.subheader("Ticket & Class")
        pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
        fare = st.number_input(
            "Ticket Fare (£)", min_value=0.0, max_value=600.0, value=15.0, step=1.0
        )
        embarked = st.selectbox(
            "Port of Embarkation",
            ["Southampton (S)", "Cherbourg (C)", "Queenstown (Q)"],
            index=0,
        )
        embarked_code = embarked.split("(")[-1].replace(")", "").strip()

    with col3:
        st.subheader("Family & Cabin")
        sibsp = st.number_input("Siblings / Spouses Aboard (SibSp)", min_value=0, max_value=8, value=0)
        parch = st.number_input("Parents / Children Aboard (Parch)", min_value=0, max_value=9, value=0)
        has_cabin = st.checkbox("Cabin Assigned?", value=False)
        cabin_deck = "Unknown"
        if has_cabin:
            cabin_deck = st.selectbox("Cabin Deck", ["A", "B", "C", "D", "E", "F"])
            cabin_code = f"{cabin_deck}123"
        else:
            cabin_code = ""

    # Construct passenger dictionary
    clean_title = "Rare" if "Rare" in title else title
    constructed_name = f"Passenger, {clean_title}. Test"

    passenger_data = {
        "PassengerId": 9999,
        "Pclass": pclass,
        "Name": constructed_name,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Ticket": "A/5 21171",
        "Fare": fare,
        "Cabin": cabin_code if has_cabin else "",
        "Embarked": embarked_code,
    }

    st.markdown("---")
    if st.button("🚀 Predict Survival Probability", type="primary"):
        with st.spinner("Calculating predictions..."):
            try:
                result = predict_single_passenger(passenger_data, config=config)
                prob = result["probability"]
                survived = result["survived"]

                st.markdown("### Prediction Result")
                res_col1, res_col2 = st.columns([1, 2])

                with res_col1:
                    if survived:
                        st.success(f"### {result['label']}")
                    else:
                        st.error(f"### {result['label']}")

                with res_col2:
                    st.metric(
                        label="Survival Probability",
                        value=f"{prob * 100:.1f}%",
                    )
                    st.progress(float(prob))

                # Contextual explanation
                st.info(
                    f"**Passenger Profile:** {clean_title}, {sex}, {age:.0f} yrs, "
                    f"Class {pclass}, Family Size {sibsp + parch + 1}, Fare £{fare:.2f}. "
                    + ("Has recorded cabin." if has_cabin else "No recorded cabin.")
                )
            except Exception as e:
                st.error(f"Error generating prediction: {e}")


def render_performance(artifacts: dict | None) -> None:
    """Render model performance metrics and CV scores."""
    st.header("📊 Model Performance & Ensemble Architecture")

    if artifacts is None:
        st.warning(
            "⚠️ No trained model found in `models/titanic_model.joblib`. "
            "Please run `python train.py` first."
        )
        return

    cv_scores = artifacts.get("cv_scores", {})
    training_time = artifacts.get("training_time", 0.0)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Training Time", f"{training_time:.2f}s")
    with col2:
        if cv_scores:
            float_scores = [v for v in cv_scores.values() if isinstance(v, float)]
            if float_scores:
                st.metric("Top Out-Of-Fold (OOF) Accuracy", f"{max(float_scores):.4f}")

    if cv_scores:
        st.subheader("Out-of-Fold (OOF) Model Accuracies")
        data = []
        for name, score in cv_scores.items():
            if isinstance(score, float):
                data.append({"Model": name, "OOF Accuracy": round(score, 4)})
            elif isinstance(score, list):
                s_arr = pd.Series(score)
                data.append({"Model": name, "OOF Accuracy": round(s_arr.mean(), 4)})

        df_scores = pd.DataFrame(data).sort_values(by="OOF Accuracy", ascending=False)
        st.dataframe(df_scores, use_container_width=True)
        st.bar_chart(df_scores.set_index("Model"))


def render_eda(train_df: pd.DataFrame | None) -> None:
    """Render exploratory data analysis visuals."""
    st.header("📈 Exploratory Data Analysis")

    if train_df is None:
        st.warning("⚠️ Training data not found in `data/train.csv`.")
        return

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Survival Rate by Gender")
        gender_surv = train_df.groupby("Sex")["Survived"].mean().reset_index()
        st.bar_chart(gender_surv.set_index("Sex"))

    with col2:
        st.subheader("Survival Rate by Passenger Class")
        class_surv = train_df.groupby("Pclass")["Survived"].mean().reset_index()
        st.bar_chart(class_surv.set_index("Pclass"))

    st.subheader("Dataset Overview")
    st.write(f"Total training passengers: {len(train_df)}")
    st.dataframe(train_df.head(10), use_container_width=True)


if __name__ == "__main__":
    main()
