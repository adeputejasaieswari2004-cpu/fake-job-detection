import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="AI-Based Fake Job Detection System",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 AI-Based Fake Job Detection System")
st.write("Check whether a job posting looks genuine or potentially fraudulent.")

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

st.subheader("Enter Job Posting Details")

title = st.text_input("Job Title")
company_profile = st.text_area("Company Profile")
description = st.text_area("Job Description")
requirements = st.text_area("Job Requirements")

if st.button("Check Job Posting"):
    if not title.strip() or not description.strip():
        st.warning("Please enter at least the Job Title and Job Description.")
    else:
        job_text = (
            title + " " +
            company_profile + " " +
            description + " " +
            requirements
        )

        text_vector = vectorizer.transform([job_text])
        prediction = model.predict(text_vector)[0]

        if int(prediction) == 1:
            st.error("⚠️ Result: Potentially Fake Job Posting")
        else:
            st.success("✅ Result: Likely Genuine Job Posting")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(text_vector)[0]
            st.write(
                f"Model confidence for this prediction: "
                f"{max(probabilities) * 100:.2f}%"
            )

        st.info(
            "This is an ML prediction, not proof that a job is fraudulent. "
            "Verify the employer and job details independently."
        )

st.divider()

st.subheader("Analyze a Job Dataset (CSV)")
uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success("Dataset uploaded successfully!")
        st.write("Rows:", data.shape[0], "| Columns:", data.shape[1])
        st.dataframe(data.head())

        required_columns = [
            "title", "company_profile", "description", "requirements"
        ]

        if all(col in data.columns for col in required_columns):
            if st.button("Analyze First 20 Jobs"):
                sample = data.head(20).copy()

                for col in required_columns:
                    sample[col] = sample[col].fillna("").astype(str)

                sample["combined_text"] = (
                    sample["title"] + " " +
                    sample["company_profile"] + " " +
                    sample["description"] + " " +
                    sample["requirements"]
                )

                vectors = vectorizer.transform(sample["combined_text"])
                predictions = model.predict(vectors)

                sample["Prediction"] = [
                    "Potentially Fake" if int(p) == 1 else "Likely Genuine"
                    for p in predictions
                ]

                st.subheader("Prediction Results")
                st.dataframe(
                    sample[["title", "Prediction"]],
                    use_container_width=True
                )

                counts = sample["Prediction"].value_counts()
                st.bar_chart(counts)
        else:
            st.warning(
                "For predictions, the CSV must contain these columns: "
                "title, company_profile, description, requirements."
            )

    except Exception as e:
        st.error(f"Could not process the CSV: {e}")
