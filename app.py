import streamlit as st
import pandas as pd
import pickle
import os

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="AI-Based Fake Job Detection System",
    page_icon="🛡️",
    layout="wide"
)

# ==========================================
# WEBSITE CSS DESIGN
# ==========================================

st.markdown("""
<style>
.stApp {
    background-color: #F0F7FF;
    color: #172554;
}

h1 {
    color: #123B70 !important;
    text-align: center;
    font-weight: bold;
}

h2, h3 {
    color: #123B70 !important;
}

p, label {
    color: #172554;
}

.stButton > button {
    background-color: #1769AA;
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 10px 22px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #0B4778;
    color: white !important;
}

[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #D5E5F5;
}

[data-testid="stFileUploader"] {
    background-color: white;
    padding: 12px;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# MAIN HEADING
# ==========================================

st.title("🛡️ AI-Based Fake Job Detection System")

st.markdown(
    "<h4 style='text-align:center;'>"
    "Detect Fake Jobs. Protect Your Career."
    "</h4>",
    unsafe_allow_html=True
)

st.markdown("---")


# ==========================================
# SESSION COUNTERS
# ==========================================

if "total_jobs" not in st.session_state:
    st.session_state.total_jobs = 0

if "fake_jobs" not in st.session_state:
    st.session_state.fake_jobs = 0

if "genuine_jobs" not in st.session_state:
    st.session_state.genuine_jobs = 0


# Dashboard placeholder: directly below main heading
dashboard_slot = st.empty()


# ==========================================
# LOAD MODEL AND VECTORIZER
# ==========================================

@st.cache_resource
def load_model_files():

    model_file = None

    possible_model_files = [
        "fake_job_model (1).pkl",
        "fake_job_model.pkl"
    ]

    for filename in possible_model_files:
        if os.path.exists(filename):
            model_file = filename
            break

    vectorizer_file = "tfidf_vectorizer.pkl"

    if model_file is None:
        return None, None, "Model file not found in repository."

    if not os.path.exists(vectorizer_file):
        return None, None, "TF-IDF vectorizer file not found."

    try:
        with open(model_file, "rb") as file:
            model = pickle.load(file)

        with open(vectorizer_file, "rb") as file:
            vectorizer = pickle.load(file)

        return model, vectorizer, None

    except Exception:
        return (
            None,
            None,
            "Could not load the model files. Please upload the "
            "original trained .pkl files from Google Colab."
        )


model, vectorizer, model_error = load_model_files()


if model_error:
    st.error("Model loading problem: " + model_error)
else:
    st.success("Detection model is ready!")


# ==========================================
# FUNCTION TO COMBINE JOB DETAILS
# ==========================================

def prepare_job_text(title, company_profile, description, requirements):

    return (
        str(title) + " " +
        str(company_profile) + " " +
        str(description) + " " +
        str(requirements)
    )


# ==========================================
# FUNCTION TO IDENTIFY WARNING SIGNS
# ==========================================

def get_reasons(title, company_profile, description, requirements):

    reasons = []

    if not str(company_profile).strip():
        reasons.append(
            "Company profile is missing. Verify the employer."
        )

    if not str(requirements).strip():
        reasons.append(
            "Job requirements are missing or unclear."
        )

    if len(str(description).strip()) < 50:
        reasons.append(
            "The job description contains limited information."
        )

    warning_phrases = [
        "registration fee",
        "pay to apply",
        "upfront payment",
        "security deposit",
        "guaranteed income",
        "processing fee",
        "pay for job"
    ]

    combined_text = (
        str(title) + " " +
        str(company_profile) + " " +
        str(description) + " " +
        str(requirements)
    ).lower()

    for phrase in warning_phrases:
        if phrase in combined_text:
            reasons.append(
                "Possible warning sign: '" + phrase +
                "' appears in the job details."
            )

    if not reasons:
        reasons.append(
            "No basic warning signs were detected. "
            "Verify the employer independently."
        )

    return reasons


# ==========================================
# INDIVIDUAL JOB DETECTION
# ==========================================

st.header("🔍 Check an Individual Job")

st.write(
    "Enter job details below to check whether the posting "
    "may be fake or genuine."
)

with st.form("job_detection_form"):

    job_title = st.text_input("Job Title")

    company_profile = st.text_area(
        "Company Profile",
        placeholder="Enter company information..."
    )

    description = st.text_area(
        "Job Description",
        placeholder="Enter the job description..."
    )

    requirements = st.text_area(
        "Job Requirements",
        placeholder="Enter job requirements..."
    )

    check_button = st.form_submit_button("🔎 Check Job")


if check_button:

    if model_error:
        st.error(
            "Prediction is unavailable until the model files are fixed."
        )

    elif not job_title.strip() and not description.strip():
        st.warning(
            "Please enter at least a job title or job description."
        )

    else:

        try:
            job_text = prepare_job_text(
                job_title,
                company_profile,
                description,
                requirements
            )

            job_vector = vectorizer.transform([job_text])

            prediction = int(model.predict(job_vector)[0])

            # This mapping assumes:
            # 1 = Fake and 0 = Genuine.
            # It must match the original training labels.

            st.session_state.total_jobs += 1

            if prediction == 1:

                st.session_state.fake_jobs += 1

                st.error("⚠️ Potentially Fake Job Posting")

                st.subheader("🚩 Possible Warning Signs")

            else:

                st.session_state.genuine_jobs += 1

                st.success("✅ Likely Genuine Job Posting")

                st.subheader("🔎 Verification Advice")

            reasons = get_reasons(
                job_title,
                company_profile,
                description,
                requirements
            )

            for reason in reasons:
                st.write("• " + reason)

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(job_vector)[0]

                confidence = max(probabilities) * 100

                st.metric(
                    "Model Prediction Confidence",
                    f"{confidence:.2f}%"
                )

            st.info(
                "This is an AI prediction, not proof of fraud. "
                "Always verify the company and never pay money "
                "to obtain a job."
            )

        except Exception:
            st.error(
                "Prediction failed. Please verify that the model "
                "and vectorizer are compatible."
            )


# ==========================================
# CSV DATASET ANALYSIS
# ==========================================

st.markdown("---")

st.header("📂 Analyse Multiple Job Postings")

st.write(
    "Upload a CSV dataset to analyse its job postings."
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        data = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully!")

        st.subheader("Dataset Preview")

        st.dataframe(data.head())

        st.write("Total rows:", data.shape[0])
        st.write("Total columns:", data.shape[1])

        required_columns = [
            "title",
            "company_profile",
            "description",
            "requirements"
        ]

        missing_columns = [
            column for column in required_columns
            if column not in data.columns
        ]

        if missing_columns:

            st.warning(
                "Required columns missing: " +
                ", ".join(missing_columns)
            )

        elif st.button("🔎 Analyse First 20 Jobs"):

            if model_error:
                st.error(
                    "Fix the model loading problem before analysis."
                )

            else:

                sample = data.head(20).copy()

                job_texts = []

                for _, row in sample.iterrows():

                    job_texts.append(
                        prepare_job_text(
                            row["title"],
                            row["company_profile"],
                            row["description"],
                            row["requirements"]
                        )
                    )

                vectors = vectorizer.transform(job_texts)

                predictions = model.predict(vectors)

                sample["Prediction"] = [
                    "Potentially Fake" if int(value) == 1
                    else "Likely Genuine"
                    for value in predictions
                ]

                fake_count = sum(
                    int(value) == 1 for value in predictions
                )

                genuine_count = sum(
                    int(value) == 0 for value in predictions
                )

                st.session_state.total_jobs += len(predictions)
                st.session_state.fake_jobs += fake_count
                st.session_state.genuine_jobs += genuine_count

                st.subheader("📋 Analysis Results")

                st.dataframe(sample)

                st.subheader("📊 Prediction Summary")

                st.bar_chart(
                    sample["Prediction"].value_counts()
                )

                st.info(
                    "Only the first 20 rows are analysed. "
                    "Clicking the button again counts those rows again."
                )

    except Exception:
        st.error(
            "Unable to read the CSV file. Please check its format."
        )


# ==========================================
# DASHBOARD DISPLAY
# ==========================================
# Render dashboard after processing predictions,
# while keeping it directly below the main heading.

with dashboard_slot.container():

    st.subheader("📊 Job Detection Dashboard")

    total = st.session_state.total_jobs
    fake = st.session_state.fake_jobs
    genuine = st.session_state.genuine_jobs

    fake_percentage = (
        fake / total * 100 if total > 0 else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Jobs Analysed", total)
    col2.metric("Fake Jobs", fake)
    col3.metric("Genuine Jobs", genuine)
    col4.metric("Fake Job Rate", f"{fake_percentage:.1f}%")


# ==========================================
# FOOTER
# ==========================================

st.markdown("---")

st.caption(
    "AI-Based Fake Job Detection System | Academic TRW Project"
)