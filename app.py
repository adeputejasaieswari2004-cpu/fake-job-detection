import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AI-Based Fake Job Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Based Fake Job Detection System")
st.write(
    "Analyze job postings using Machine Learning "
    "and identify possible warning signs."
)

@st.cache_resource
def load_models():
    model = joblib.load("fake_job_model (1).pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_models()
    st.success("Detection model loaded successfully!")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()


def get_reasons(title, company, description, requirements):
    reasons = []

    if not company.strip():
        reasons.append(
            "Company profile is missing. Verify the employer independently."
        )

    if not requirements.strip():
        reasons.append(
            "Job requirements are missing. Check the complete job details."
        )

    if len(description.strip()) < 50:
        reasons.append(
            "The job description is very short. Ask for more details."
        )

    warning_phrases = [
        "registration fee",
        "pay to apply",
        "upfront payment",
        "security deposit",
        "guaranteed income"
    ]

    combined_text = (
        title + " " + company + " " +
        description + " " + requirements
    ).lower()

    for phrase in warning_phrases:
        if phrase in combined_text:
            reasons.append(
                f"The posting mentions '{phrase}'. Verify carefully "
                "and never pay money to obtain a job."
            )

    if not reasons:
        reasons.append(
            "No warning signs from these basic checks were identified. "
            "Still verify the employer and job offer."
        )

    return reasons


st.header("1. Check an Individual Job Posting")

title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Job Requirements")

if st.button("Check Job Posting"):
    if not title.strip() or not description.strip():
        st.warning("Please enter the Job Title and Job Description.")
    else:
        job_text = (
            title + " " +
            company_profile + " " +
            description + " " +
            requirements
        )

        text_vector = vectorizer.transform([job_text])
        prediction = model.predict(text_vector)[0]

        st.subheader("Prediction Result")

        if int(prediction) == 1:
            st.error("⚠️ Potentially Fake Job Posting")
        else:
            st.success("✅ Likely Genuine Job Posting")

        st.subheader("Why should you check this job?")

        reasons = get_reasons(
            title, company_profile, description, requirements
        )

        for reason in reasons:
            st.write("⚠️", reason)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vector)[0]
            confidence = max(probabilities) * 100

            st.metric("Model Prediction Confidence", f"{confidence:.2f}%")

        st.info(
            "These are preliminary warning signs, not proof of fraud. "
            "The model can make mistakes. Verify the employer independently."
        )


st.divider()

st.header("2. Analyze Job Postings from a CSV Dataset")

uploaded_file = st.file_uploader(
    "Upload a job postings CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")
        st.write("Total rows:", len(data))
        st.write("Total columns:", len(data.columns))
        st.dataframe(data.head())

        required_columns = [
            "title",
            "company_profile",
            "description",
            "requirements"
        ]

        if all(col in data.columns for col in required_columns):

            if st.button("Analyze First 20 Jobs"):
                sample = data.head(20).copy()

                for col in required_columns:
                    sample[col] = (
                        sample[col].fillna("").astype(str)
                    )

                sample["combined_text"] = (
                    sample["title"] + " " +
                    sample["company_profile"] + " " +
                    sample["description"] + " " +
                    sample["requirements"]
                )

                vectors = vectorizer.transform(
                    sample["combined_text"]
                )

                predictions = model.predict(vectors)

                sample["Prediction"] = [
                    "Potentially Fake" if int(p) == 1
                    else "Likely Genuine"
                    for p in predictions
                ]

                sample["Warning Signs"] = [
                    "; ".join(
                        get_reasons(
                            row["title"],
                            row["company_profile"],
                            row["description"],
                            row["requirements"]
                        )
                    )
                    for _, row in sample.iterrows()
                ]

                st.subheader("Prediction Results")

                st.dataframe(
                    sample[
                        ["title", "Prediction", "Warning Signs"]
                    ],
                    use_container_width=True
                )

                st.subheader("Prediction Summary")

                counts = sample["Prediction"].value_counts()
                st.bar_chart(counts)

        else:
            st.warning(
                "CSV must contain title, company_profile, "
                "description and requirements columns."
            )

    except Exception as e:
        st.error(f"Could not process the dataset: {e}")