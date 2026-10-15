import streamlit as st
import pandas as pd
import joblib

# ---------------- PAGE CONFIGURATION ----------------

st.set_page_config(
    page_title="AI-Based Fake Job Detection System",
    page_icon="🔍",
    layout="wide"
)

# ---------------- CSS DESIGN ----------------

st.markdown("""
<style>

/* Main Background */
.stApp {
    background-color: #F4F7FB !important;
}

/* Main Text Visibility */
.stApp,
.stApp p,
.stApp label,
.stApp span,
.stApp div,
.stApp li {
    color: #172B4D;
}

/* Main Heading */
.main-title {
    background: linear-gradient(120deg, #123C69, #2471A3);
    padding: 28px 18px;
    border-radius: 16px;
    text-align: center;
    margin-bottom: 20px;
}

.main-title h1 {
    color: #FFFFFF !important;
    font-size: 30px;
    font-weight: bold;
    margin-bottom: 12px;
}

.main-title p {
    color: #FFFFFF !important;
    font-size: 16px;
    margin-bottom: 6px;
}

/* All Section Headings */
h1, h2, h3, h4 {
    color: #123C69 !important;
    font-weight: bold !important;
}

/* Dashboard Metric Cards */
div[data-testid="stMetric"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D6E4F0;
    border-radius: 12px;
    padding: 18px;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.05);
}

/* Dashboard Labels and Values */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label *,
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] *,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] *,
div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] * {
    color: #123C69 !important;
}

/* Input Fields */
.stTextInput input,
.stTextArea textarea,
.stNumberInput input {
    background-color: #FFFFFF !important;
    color: #172B4D !important;
    border: 1px solid #AABED0 !important;
    border-radius: 8px !important;
    caret-color: #123C69 !important;
}

/* Input Placeholder */
.stTextInput input::placeholder,
.stTextArea textarea::placeholder {
    color: #66788A !important;
    opacity: 1 !important;
}

/* Input Focus */
.stTextInput input:focus,
.stTextArea textarea:focus,
.stNumberInput input:focus {
    border: 2px solid #2471A3 !important;
    box-shadow: 0 0 4px rgba(36, 113, 163, 0.25);
}

/* Buttons */
.stButton > button {
    background-color: #1F618D !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    padding: 10px 20px !important;
    font-weight: bold !important;
}

/* Button Text */
.stButton > button p,
.stButton > button span {
    color: #FFFFFF !important;
}

/* Button Hover */
.stButton > button:hover {
    background-color: #154360 !important;
    border: 1px solid #154360 !important;
}

/* Download Button */
.stDownloadButton > button {
    background-color: #117864 !important;
    color: #FFFFFF !important;
    border-radius: 9px !important;
    border: none !important;
}

.stDownloadButton > button p,
.stDownloadButton > button span {
    color: #FFFFFF !important;
}

/* File Uploader */
div[data-testid="stFileUploader"] {
    background-color: #FFFFFF !important;
    border: 1px solid #D6E4F0;
    border-radius: 10px;
    padding: 15px;
}

/* File Uploader Text */
div[data-testid="stFileUploader"] * {
    color: #172B4D;
}

/* Dataframe */
div[data-testid="stDataFrame"] {
    background-color: #FFFFFF !important;
}

/* Tabs */
.stTabs [data-baseweb="tab"] {
    color: #123C69 !important;
}

/* Dividers */
hr {
    border-color: #C8D8E8 !important;
}

/* Alerts */
div[data-testid="stAlert"] p {
    color: #172B4D !important;
}

/* Footer */
.footer-text {
    color: #52677D !important;
    text-align: center;
    font-size: 13px;
    padding: 12px;
}

/* Mobile Responsive */
@media (max-width: 768px) {
    .main-title {
        padding: 20px 12px;
    }

    .main-title h1 {
        font-size: 23px;
    }

    .main-title p {
        font-size: 14px;
    }

    div[data-testid="stMetric"] {
        padding: 10px;
    }
}

</style>
""", unsafe_allow_html=True)


# ---------------- HEADER ----------------

st.markdown("""
<div class="main-title">
    <h1>🔍 AI-Based Fake Job Detection System</h1>
    <p>Detect Fake Jobs. Protect Your Career.</p>
    <p>Analyze job postings using Machine Learning
    and identify possible warning signs.</p>
</div>
""", unsafe_allow_html=True)


# ---------------- DASHBOARD POSITION ----------------

dashboard_slot = st.empty()


# ---------------- SESSION STATE ----------------

if "total_jobs" not in st.session_state:
    st.session_state.total_jobs = 0

if "fake_jobs" not in st.session_state:
    st.session_state.fake_jobs = 0

if "genuine_jobs" not in st.session_state:
    st.session_state.genuine_jobs = 0


# ---------------- LOAD MODEL ----------------

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
    st.info(
        "Check whether both model files are available "
        "in your GitHub repository."
    )
    st.stop()


# ---------------- WARNING REASONS ----------------

def get_reasons(title, company, description, requirements):

    reasons = []

    if not company.strip():
        reasons.append(
            "Company profile is missing. "
            "Verify the employer independently."
        )

    if not requirements.strip():
        reasons.append(
            "Job requirements are missing. "
            "Check the complete job details."
        )

    if len(description.strip()) < 50:
        reasons.append(
            "The job description is very short. "
            "Ask for more details."
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
                f"The posting mentions '{phrase}'. "
                "Verify carefully and never pay money "
                "to obtain a job."
            )

    if not reasons:
        reasons.append(
            "No warning signs from these basic checks "
            "were identified. Still verify the employer "
            "and job offer."
        )

    return reasons


# ---------------- DASHBOARD ----------------

def show_dashboard():

    with dashboard_slot.container():

        st.subheader("📊 Job Detection Dashboard")

        col1, col2, col3, col4 = st.columns(4)

        total = st.session_state.total_jobs
        fake = st.session_state.fake_jobs
        genuine = st.session_state.genuine_jobs

        rate = fake / total * 100 if total > 0 else 0

        col1.metric("Total Jobs Analysed", total)
        col2.metric("Fake Jobs", fake)
        col3.metric("Genuine Jobs", genuine)
        col4.metric("Fake Job Rate", f"{rate:.2f}%")


show_dashboard()

st.divider()


# ==================================================
# 1. INDIVIDUAL JOB POSTING
# ==================================================

st.header("1. Check an Individual Job Posting")

st.write(
    "Enter job details below to check the posting "
    "using the trained Machine Learning model."
)

title = st.text_input("Job Title")

company_profile = st.text_area("Company Profile")

description = st.text_area("Job Description")

requirements = st.text_area("Job Requirements")


if st.button("🔍 Check Job Posting"):

    if not title.strip() or not description.strip():

        st.warning(
            "Please enter the Job Title and Job Description."
        )

    else:

        job_text = (
            title + " " + company_profile + " " +
            description + " " + requirements
        )

        text_vector = vectorizer.transform([job_text])

        prediction = model.predict(text_vector)[0]

        st.subheader("Prediction Result")

        st.session_state.total_jobs += 1

        if int(prediction) == 1:

            st.session_state.fake_jobs += 1

            st.error("⚠️ Potentially Fake Job Posting")

        else:

            st.session_state.genuine_jobs += 1

            st.success("✅ Likely Genuine Job Posting")

        # ---------------- CONFIDENCE ----------------

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                text_vector
            )[0]

            confidence = max(probabilities) * 100

            st.metric(
                "Model Prediction Confidence",
                f"{confidence:.2f}%"
            )

        # ---------------- WARNING REASONS ----------------

        st.subheader("⚠️ Why Should You Check This Job?")

        reasons = get_reasons(
            title,
            company_profile,
            description,
            requirements
        )

        for reason in reasons:
            st.write("⚠️", reason)

        st.info(
            "These are preliminary warning signs, not proof "
            "of fraud. The model can make mistakes. "
            "Verify the employer independently."
        )

        # Refresh dashboard metrics
        show_dashboard()


st.divider()


# ==================================================
# 2. CSV DATASET ANALYSIS
# ==================================================

st.header("2. Analyze Job Postings from a CSV Dataset")

st.write(
    "Upload a CSV file containing job postings "
    "to analyze multiple jobs."
)

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

        st.subheader("Dataset Preview")

        st.dataframe(
            data.head(),
            use_container_width=True
        )

        required_columns = [
            "title",
            "company_profile",
            "description",
            "requirements"
        ]

        if all(col in data.columns for col in required_columns):

            st.subheader("Select Number of Jobs")

            number_of_jobs = st.number_input(
                "How many jobs do you want to analyze?",
                min_value=1,
                max_value=max(1, len(data)),
                value=min(20, max(1, len(data))),
                step=1
            )

            if st.button("📊 Analyze Jobs"):

                sample = data.head(
                    int(number_of_jobs)
                ).copy()

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

                # ---------------- RESULTS ----------------

                st.subheader("Prediction Results")

                st.dataframe(
                    sample[
                        ["title", "Prediction", "Warning Signs"]
                    ],
                    use_container_width=True
                )

                # ---------------- SUMMARY ----------------

                st.subheader("📈 Prediction Summary")

                counts = sample["Prediction"].value_counts()

                fake_count = int(
                    (sample["Prediction"] == "Potentially Fake").sum()
                )

                genuine_count = int(
                    (sample["Prediction"] == "Likely Genuine").sum()
                )

                col1, col2, col3 = st.columns(3)

                col1.metric("Jobs Analyzed", len(sample))
                col2.metric("Potentially Fake", fake_count)
                col3.metric("Likely Genuine", genuine_count)

                st.bar_chart(counts)

                # ---------------- DOWNLOAD RESULTS ----------------

                st.subheader("Download Analysis Results")

                download_data = sample.drop(
                    columns=["combined_text"]
                )

                csv_data = download_data.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_data,
                    file_name="fake_job_detection_results.csv",
                    mime="text/csv"
                )

                st.info(
                    "CSV predictions are based on the trained "
                    "Machine Learning model. Warning signs are "
                    "preliminary checks and do not prove fraud."
                )

        else:

            st.warning(
                "CSV must contain these columns: "
                "title, company_profile, description "
                "and requirements."
            )

    except Exception as e:

        st.error(f"Could not process the dataset: {e}")


# ---------------- FOOTER ----------------

st.divider()

st.markdown("""
<div class="footer-text">
    AI-Based Fake Job Detection System |
    Machine Learning Project |
    Always verify job offers independently.
</div>
""", unsafe_allow_html=True)