import streamlit as st
import pandas as pd
import pickle
import os
import re
import html
from collections import Counter

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI-Based Fake Job Detection System",
    page_icon="🛡️",
    layout="wide"
)

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #F4F7FB;
    color: #172B4D;
}

header[data-testid="stHeader"] {
    background-color: #F4F7FB;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: #123C69 !important;
}

p, label, li {
    color: #172B4D;
}

.main-title {
    background: linear-gradient(90deg, #123C69, #2471A3);
    color: white !important;
    padding: 25px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 20px;
}

.main-title h1 {
    color: white !important;
    margin-bottom: 8px;
}

.main-title p {
    color: white !important;
    font-size: 17px;
}

.section-card {
    background-color: #FFFFFF;
    border: 1px solid #D6E4F0;
    border-radius: 14px;
    padding: 20px;
    margin-bottom: 18px;
}

.metric-card {
    background-color: #FFFFFF;
    border: 1px solid #D6E4F0;
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}

.metric-card h3 {
    font-size: 15px;
    margin-bottom: 8px;
}

.metric-card h2 {
    font-size: 28px;
    margin: 0;
}

.fake-result {
    background-color: #FDECEC;
    border-left: 6px solid #D93025;
    padding: 18px;
    border-radius: 10px;
    color: #7A1712 !important;
}

.real-result {
    background-color: #E8F7EE;
    border-left: 6px solid #18864B;
    padding: 18px;
    border-radius: 10px;
    color: #125B35 !important;
}

.info-box {
    background-color: #EAF3FC;
    border-left: 5px solid #2471A3;
    padding: 15px;
    border-radius: 8px;
    color: #123C69 !important;
}

.warning-box {
    background-color: #FFF4DD;
    border-left: 5px solid #D99A00;
    padding: 15px;
    border-radius: 8px;
    color: #704D00 !important;
}

/* CSV FILE UPLOADER */

div[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 2px solid #2471A3 !important;
    border-radius: 12px !important;
    padding: 20px !important;
}

/* Make uploader text visible */

div[data-testid="stFileUploader"] label,
div[data-testid="stFileUploader"] label *,
div[data-testid="stFileUploader"] section,
div[data-testid="stFileUploader"] section *,
div[data-testid="stFileUploader"] small,
div[data-testid="stFileUploader"] span,
div[data-testid="stFileUploader"] p,
div[data-testid="stFileUploader"] [data-testid="stMarkdownContainer"] * {
    color: #172B4D !important;
    opacity: 1 !important;
}

/* Choose File Button */

div[data-testid="stFileUploader"] button {
    background-color: #E8F2FC !important;
    color: #123C69 !important;
    border: 1px solid #2471A3 !important;
    border-radius: 8px !important;
}

div[data-testid="stFileUploader"] button * {
    color: #123C69 !important;
}

div[data-testid="stFileUploader"] button:hover {
    background-color: #D4E6F7 !important;
}

/* Upload instructions */

div[data-testid="stFileUploaderDropzoneInstructions"] * {
    color: #172B4D !important;
    opacity: 1 !important;
}

/* Buttons */

.stButton > button,
.stDownloadButton > button {
    background-color: #2471A3;
    color: white !important;
    border-radius: 9px;
    border: none;
    padding: 10px 18px;
    font-weight: bold;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background-color: #123C69;
    color: white !important;
}

/* Inputs */

.stTextInput input,
.stTextArea textarea,
.stSelectbox div[data-baseweb="select"] {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
}

/* Dataframes */

div[data-testid="stDataFrame"] {
    border: 1px solid #D6E4F0;
    border-radius: 10px;
}

/* Footer */

.footer {
    text-align: center;
    color: #526579;
    padding: 20px;
    margin-top: 25px;
    border-top: 1px solid #D6E4F0;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# LOAD TRAINED MODEL AND VECTORIZER
# --------------------------------------------------

MODEL_PATH = "fake_job_model (1).pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"

@st.cache_resource
def load_model_files():
    if not os.path.exists(MODEL_PATH):
        return None, None, (
            f"Model file not found: {MODEL_PATH}. "
            "Please upload the trained model file to your GitHub repository."
        )

    if not os.path.exists(VECTORIZER_PATH):
        return None, None, (
            f"Vectorizer file not found: {VECTORIZER_PATH}. "
            "Please upload the vectorizer file to your GitHub repository."
        )

    try:
        with open(MODEL_PATH, "rb") as file:
            model = pickle.load(file)

        with open(VECTORIZER_PATH, "rb") as file:
            vectorizer = pickle.load(file)

        return model, vectorizer, None

    except Exception as error:
        return None, None, str(error)


model, vectorizer, model_error = load_model_files()


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "total_checked" not in st.session_state:
    st.session_state.total_checked = 0

if "fake_count" not in st.session_state:
    st.session_state.fake_count = 0

if "genuine_count" not in st.session_state:
    st.session_state.genuine_count = 0


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown("""
<div class="main-title">
    <h1>🛡️ AI-Based Fake Job Detection System</h1>
    <p>Detect Fake Jobs. Protect Your Career.</p>
</div>
""", unsafe_allow_html=True)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("🛡️ Navigation")

page = st.sidebar.radio(
    "Select a Section",
    [
        "Dashboard",
        "Check Individual Job",
        "Analyze CSV Dataset",
        "About Project"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This system uses a machine-learning model trained "
    "on labelled job-posting data."
)

if model_error:
    st.sidebar.error("Model files are not ready.")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def safe_text(value):
    if pd.isna(value):
        return ""
    return str(value)


def combine_job_text(job):
    fields = [
        "title",
        "company_profile",
        "description",
        "requirements",
        "benefits",
        "employment_type",
        "required_experience",
        "required_education",
        "industry",
        "function",
        "location",
        "department",
        "salary_range"
    ]

    parts = []

    for field in fields:
        if field in job.index:
            value = safe_text(job[field])
            if value.strip():
                parts.append(value)

    return " ".join(parts)


def detect_warning_reasons(job):
    reasons = []

    company_profile = safe_text(
        job.get("company_profile", "")
    ).strip()

    description = safe_text(
        job.get("description", "")
    ).strip()

    requirements = safe_text(
        job.get("requirements", "")
    ).strip()

    benefits = safe_text(
        job.get("benefits", "")
    ).strip()

    logo = safe_text(
        job.get("has_company_logo", "")
    ).strip()

    questions = safe_text(
        job.get("has_questions", "")
    ).strip()

    salary = safe_text(
        job.get("salary_range", "")
    ).strip()

    title = safe_text(
        job.get("title", "")
    ).lower()

    # Missing information checks

    if not company_profile:
        reasons.append(
            "Company profile information is missing."
        )

    if not description:
        reasons.append(
            "Job description is missing."
        )

    if not requirements:
        reasons.append(
            "Job requirements are not provided."
        )

    if not benefits:
        reasons.append(
            "Benefits information is missing."
        )

    if logo in ["0", "0.0", "False", "false"]:
        reasons.append(
            "The posting indicates that a company logo is absent."
        )

    # Suspicious wording checks

    suspicious_phrases = [
        "pay upfront",
        "registration fee",
        "processing fee",
        "pay to apply",
        "no experience required",
        "earn money fast",
        "guaranteed income",
        "send money",
        "wire transfer",
        "bank account details",
        "instant earnings",
        "work from home and earn"
    ]

    combined_text = (
        title + " " + description + " " + requirements
    ).lower()

    for phrase in suspicious_phrases:
        if phrase in combined_text:
            reasons.append(
                f"Potential warning phrase detected: '{phrase}'."
            )

    if salary:
        if re.search(r"\$\s*0\b|0\s*-\s*0", salary):
            reasons.append(
                "The listed salary range may require verification."
            )

    if not reasons:
        reasons.append(
            "No specific warning indicator was identified by "
            "these basic text checks. Verify the employer independently."
        )

    return reasons


def predict_job(job):
    job_text = combine_job_text(job)

    if not job_text.strip():
        return None, None, [
            "Insufficient job information to make a prediction."
        ]

    transformed_text = vectorizer.transform([job_text])

    prediction = int(model.predict(transformed_text)[0])

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(transformed_text)[0]
        confidence = float(max(probabilities)) * 100

    reasons = detect_warning_reasons(job)

    return prediction, confidence, reasons


def get_job_result(prediction):
    if prediction == 1:
        return "Potential Fake Job"
    return "Likely Genuine Job"


# --------------------------------------------------
# DASHBOARD
# --------------------------------------------------

if page == "Dashboard":

    st.subheader("📊 Job Detection Dashboard")

    st.write(
        "View the job-checking activity performed during this session."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Jobs Checked</h3>
            <h2>{st.session_state.total_checked}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Potential Fake Jobs</h3>
            <h2>{st.session_state.fake_count}</h2>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Likely Genuine Jobs</h3>
            <h2>{st.session_state.genuine_count}</h2>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.subheader("🔍 How to Use the System")

    st.markdown("""
    <div class="section-card">
        <b>Step 1:</b> Select Check Individual Job to check one job posting.<br><br>
        <b>Step 2:</b> Enter the available job details.<br><br>
        <b>Step 3:</b> View the model prediction and possible warning indicators.<br><br>
        <b>Step 4:</b> Select Analyze CSV Dataset to analyze multiple job postings.<br><br>
        <b>Step 5:</b> Download the analysis results as a CSV file.
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Dashboard counts are session-based. They reset when the session "
        "is reset and are not a permanent database of all visitors."
    )


# --------------------------------------------------
# INDIVIDUAL JOB CHECK
# --------------------------------------------------

elif page == "Check Individual Job":

    st.subheader("🔎 Check an Individual Job Posting")

    st.write(
        "Enter the job-posting details to receive a model prediction "
        "and possible warning indicators."
    )

    with st.form("individual_job_form"):

        title = st.text_input("Job Title")

        location = st.text_input("Job Location")

        department = st.text_input("Department")

        company_profile = st.text_area(
            "Company Profile",
            height=100
        )

        description = st.text_area(
            "Job Description",
            height=150
        )

        requirements = st.text_area(
            "Job Requirements",
            height=100
        )

        benefits = st.text_area(
            "Benefits",
            height=100
        )

        salary_range = st.text_input("Salary Range")

        employment_type = st.selectbox(
            "Employment Type",
            [
                "",
                "Full-time",
                "Part-time",
                "Contract",
                "Temporary",
                "Other"
            ]
        )

        required_experience = st.selectbox(
            "Required Experience",
            [
                "",
                "Internship",
                "Entry level",
                "Associate",
                "Mid-Senior level",
                "Director",
                "Executive"
            ]
        )

        required_education = st.text_input(
            "Required Education"
        )

        industry = st.text_input("Industry")

        function = st.text_input("Job Function")

        has_company_logo = st.selectbox(
            "Company Logo Available?",
            ["Yes", "No", "Unknown"]
        )

        has_questions = st.selectbox(
            "Screening Questions Available?",
            ["Yes", "No", "Unknown"]
        )

        submit_job = st.form_submit_button(
            "🔍 Analyze Job Posting"
        )

    if submit_job:

        if model_error:
            st.error(
                "The model could not be loaded. "
                "Check that both model files exist in the repository."
            )

        elif not title.strip() and not description.strip():
            st.warning(
                "Please enter at least a job title or job description."
            )

        else:

            job_data = pd.Series({
                "title": title,
                "location": location,
                "department": department,
                "company_profile": company_profile,
                "description": description,
                "requirements": requirements,
                "benefits": benefits,
                "salary_range": salary_range,
                "employment_type": employment_type,
                "required_experience": required_experience,
                "required_education": required_education,
                "industry": industry,
                "function": function,
                "has_company_logo": (
                    "1" if has_company_logo == "Yes"
                    else "0" if has_company_logo == "No"
                    else ""
                ),
                "has_questions": (
                    "1" if has_questions == "Yes"
                    else "0" if has_questions == "No"
                    else ""
                )
            })

            with st.spinner("Analyzing the job posting..."):

                prediction, confidence, reasons = predict_job(
                    job_data
                )

            if prediction is None:

                st.warning(
                    "Not enough information to analyze this job."
                )

            else:

                st.session_state.total_checked += 1

                if prediction == 1:
                    st.session_state.fake_count += 1
                else:
                    st.session_state.genuine_count += 1

                st.markdown("---")
                st.subheader("📋 Prediction Result")

                if prediction == 1:

                    st.markdown("""
                    <div class="fake-result">
                        <h3>⚠️ Potential Fake Job</h3>
                        This model predicts that the job posting may be fraudulent.
                        Verify the employer and job details before taking action.
                    </div>
                    """, unsafe_allow_html=True)

                else:

                    st.markdown("""
                    <div class="real-result">
                        <h3>✅ Likely Genuine Job</h3>
                        This model predicts that the job posting is likely genuine.
                        This result does not guarantee that the job is safe.
                    </div>
                    """, unsafe_allow_html=True)

                if confidence is not None:

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.caption(
                        "This is the model's prediction confidence, "
                        "not a guarantee that the result is correct."
                    )

                st.subheader("🚩 Possible Warning Indicators")

                for reason in reasons:
                    st.write(f"• {reason}")

                st.markdown("""
                <div class="warning-box">
                    <b>Important:</b> Warning indicators are simple checks
                    based on available text and fields. They are not proof
                    of fraud. Confirm job details through official company
                    channels and never pay money to obtain a job.
                </div>
                """, unsafe_allow_html=True)


# --------------------------------------------------
# CSV DATASET ANALYSIS
# --------------------------------------------------

elif page == "Analyze CSV Dataset":

    st.subheader("2. Analyze Job Postings from a CSV Dataset")

    st.write(
        "Upload a CSV file containing job postings to analyze multiple jobs."
    )

    st.markdown("""
    <div class="info-box">
        <b>CSV Upload:</b> Select a CSV file containing job postings.
        The dataset should include job information such as title,
        description, requirements, or company profile.
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a job postings CSV file",
        type=["csv"],
        help="Choose a CSV file containing job postings."
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            st.success("Dataset uploaded successfully!")

            st.write(
                f"**Total Rows:** {data.shape[0]}  |  "
                f"**Total Columns:** {data.shape[1]}"
            )

            st.subheader("👀 Dataset Preview")

            st.dataframe(
                data.head(),
                use_container_width=True
            )

            st.markdown("---")

            st.subheader("⚙️ Analysis Settings")

            max_jobs = st.number_input(
                "Number of jobs to analyze",
                min_value=1,
                max_value=max(1, len(data)),
                value=min(100, max(1, len(data))),
                step=1
            )

            if st.button("🚀 Analyze Job Postings"):

                if model_error:

                    st.error(
                        "The model could not be loaded. "
                        "Please check the model files in GitHub."
                    )

                else:

                    total_to_analyze = min(
                        int(max_jobs),
                        len(data)
                    )

                    results = []
