import streamlit as st
import pandas as pd
import pickle
import os

# --------------------------------
# PAGE CONFIGURATION
# --------------------------------

st.set_page_config(
    page_title="AI-Based Fake Job Detection System",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------
# CUSTOM WEBSITE DESIGN
# --------------------------------

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

p, label, span, div {
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

input, textarea {
    background-color: white !important;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------
# LOAD TRAINED MODEL AND VECTORIZER
# --------------------------------

@st.cache_resource
def load_files():

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
        return None, None, "Trained model file not found."

    if not os.path.exists(vectorizer_file):
        return None, None, "TF-IDF vectorizer file not found."

    try:
        with open(model_file, "rb") as f:
            model = pickle.load(f)

        with open(vectorizer_file, "rb") as f:
            vectorizer = pickle.load(f)

        return model, vectorizer, None

    except Exception as e:
        return None, None, str(e)


model, vectorizer, error = load_files()


# --------------------------------
# MAIN WEBSITE HEADING
# --------------------------------

st.title("🛡️ AI-Based Fake Job Detection System")

st.markdown(
    "<h4 style='text-align:center;'>"
    "Detect Fake Jobs. Protect Your Career."
    "</h4>",
    unsafe_allow_html=True
)

st.markdown("---")


# --------------------------------
# DASHBOARD COUNTERS
# --------------------------------

if "total_jobs" not in st.session_state:
    st.session_state.total_jobs = 0

if "fake_jobs" not in st.session_state:
    st.session_state.fake_jobs = 0

if "genuine_jobs" not in st.session_state:
    st.session_state.genuine_jobs = 0


# Reserve dashboard space directly below the main heading.
# It will be filled after predictions are processed.

dashboard_slot = st.empty()


# --------------------------------
# CHECK MODEL STATUS
# --------------------------------

if error:
    st.error("Model loading problem: " + error)

else:
    st.success("Detection model is ready!")


# --------------------------------
# FUNCTION: COMBINE JOB DETAILS
# --------------------------------

def prepare_job_text(title, company_profile, description, requirements):

    text = (
        str(title) + " " +
        str(company_profile) + " " +
        str(description) + " " +
        str(requirements)
    )

    return text


# --------------------------------
# FUNCTION: EXPLAIN WARNING SIGNS
# --------------------------------

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
            "Job description contains very little information."
        )

    warning_phrases = [
        "registration fee",
        "pay to apply",
        "upfront payment",
        "security deposit",
        "guaranteed income",
        "pay for job",
        "processing fee"
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
            "No warning signs were detected by the basic text checks. "
            "Still verify the employer independently."
        )

    return reasons


# --------------------------------
# INDIVIDUAL JOB DETECTION
# --------------------------------

st.header("🔍 Check an Individual Job")

st.write(
    "Enter the job details below to check whether the posting "
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
        placeholder="Enter the complete job description..."
    )

    requirements = st.text_area(
        "Job Requirements",
        placeholder="Enter job requirements..."
    )

    check_button = st.form_submit_button("🔎 Check Job")


if check_button:

    if error:
        st.error("Please fix the model loading problem first.")

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

            prediction = model.predict(job_vector)[0]

            # Assumption: 1 = Fake, 0 = Genuine
            # This must match the labels used during model training.

            st.session_state.total_jobs += 1

            if int(prediction) == 1:

                st.session_state.fake_jobs += 1

                st.error("⚠️ Potentially Fake Job Posting")

                st.subheader("🚩 Possible Warning Signs")

                reasons = get_reasons(
                    job_title,
                    company_profile,
                    description,
                    requirements
                )

                for reason in reasons:
                    st.write("• " + reason)

            else:

                st.session_state.genuine_jobs += 1

                st.success("✅ Likely Genuine Job Posting")

                st.subheader("🔎 Verification Advice")

                st.write(
                    "The model classified this posting as likely genuine. "
                    "However, this does not guarantee that the job is safe."
                )

                reasons = get_reasons(
                    job_title,
                    company_profile,
                    description,
                    requirements
                )

                for reason in reasons:
                    st.write("• " + reason)

            # Show confidence when supported by the model.

            if hasattr(model, "predict_proba"):

                probabilities = model.predict_proba(job_vector)[0]

                confidence = max(probabilities) * 100

                st.metric(
                    "Model Prediction Confidence",
                    f"{confidence:.2f}%"
                )

            st.info(
                "Important: This is an AI prediction, not proof of fraud. "
                "Verify the company and never pay money to obtain a job."
            )

        except Exception as e:
            st.error(
                "Prediction failed. Please check the model and "
                "vectorizer compatibility."
            )

            st.caption(str(e))


# --------------------------------
# CSV DATASET ANALYSIS
# --------------------------------

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
            col for col in required_columns
            if col not in data.columns
        ]

        if missing_columns:

            st.warning(
                "Required columns missing: " +
                ", ".join(missing_columns)
            )

        else:

            if st.button("🔎 Analyse First 20 Jobs"):

                if error:
                    st.error(
                        "Model loading problem. "
                        "Please check the model files."
                    )

                else:

                    sample = data.head(20).copy()

                    job_texts = []

                    for _, row in sample.iterrows():

                        text = prepare_job_text(
                            row["title"],
                            row["company_profile"],
                            row["description"],
                            row["requirements"]
                        )

                        job_texts.append(text)

                    vectors = vectorizer.transform(job_texts)

                    predictions = model.predict(vectors)

                    sample["Prediction"] = [
                        "Potentially Fake"
                        if int(p) == 1
                        else "Likely Genuine"
                        for p in predictions
                    ]

                    # Update dashboard counters.

                    fake_count = sum(
                        int(p) == 1 for p in predictions
                    )

                    genuine_count = sum(
                        int(p) == 0 for p in predictions
                    )

                    st.session_state.total_jobs += len(predictions)

                    st.session_state.fake_jobs += fake_count

                    st.session_state.genuine_jobs += genuine_count

                    st.subheader("📋 Analysis Results")

                    st.dataframe(sample)

                    st.subheader("📊 Prediction Summary")

                    summary = pd.Series(
                        sample["Prediction"]
                    ).value_counts()

                    st.bar_chart(summary)

                    st.info(
                        "The analysis includes the first 20 rows only. "
                        "Each time you click the analysis button, "
                        "those rows are counted again."
                    )

    except Exception as e:

        st.error("Unable to analyse the CSV file.")

        st.caption(str(e))


# --------------------------------
# DASHBOARD DISPLAY
# --------------------------------
# This section runs after the predictions.
# The dashboard appears below the main website heading.

with dashboard_slot.container():

    st.subheader("📊 Job Detection Dashboard")

    total = st.session_state.total_jobs

    fake = st.session_state.fake_jobs

    genuine = st.session_state.genuine_jobs

    fake_percentage = (
        fake / total * 100
        if total > 0
        else 0
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Jobs Analysed",
        total
    )

    col2.metric(
        "Fake Jobs",
        fake
    )

    col3.metric(
        "Genuine Jobs",
        genuine
    )

    col4.metric(
        "Fake Job Rate",
        f"{fake_percentage:.1f}%"
    )


# --------------------------------
# FOOTER
# --------------------------------

st.markdown("---")

st.caption(
    "AI-Based Fake Job Detection System | "
    "Academic TRW Project | "
    "Predictions may be incorrect; always verify job offers."
)