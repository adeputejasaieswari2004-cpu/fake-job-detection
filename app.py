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
    .stApp {
        background-color: #f4f7fb;
    }

    .main-title {
        background: linear-gradient(90deg, #123c69, #2471a3);
        padding: 25px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 10px;
    }

    .main-title h1 {
        color: white;
        font-size: 30px;
    }

    .main-title p {
        color: #eaf2f8;
        font-size: 16px;
    }

    .section-title {
        color: #123c69;
        font-weight: bold;
    }

    div[data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #d6e4f0;
        padding: 18px;
        border-radius: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.05);
    }

    div.stButton > button {
        background-color: #1f618d;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }

    div.stButton > button:hover {
        background-color: #154360;
        color: white;
    }

    div[data-testid="stFileUploader"] {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
    }

    h2, h3 {
        color: #123c69;
    }

    @media (max-width: 768px) {
        .main-title h1 {
            font-size: 23px;
        }

        .main-title p {
            font-size: 14px;
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

# Dashboard heading tarvatha display avuthundi.
# Migatha sections dashboard kindha untayi.

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
        title + " " +
        company + " " +
        description + " " +
        requirements
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

with dashboard_slot.container():

    st.subheader("📊 Job Detection Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    fake_rate = (
        st.session_state.fake_jobs
        / st.session_state.total_jobs * 100
        if st.session_state.total_jobs > 0
        else 0
    )

    col1.metric(
        "Total Jobs Analysed",
        st.session_state.total_jobs
    )

    col2.metric(
        "Fake Jobs",
        st.session_state.fake_jobs
    )

    col3.metric(
        "Genuine Jobs",
        st.session_state.genuine_jobs
    )

    col4.metric(
        "Fake Job Rate",
        f"{fake_rate:.2f}%"
    )

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
            title + " " +
            company_profile + " " +
            description + " " +
            requirements
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

        # ---------------- REFRESH DASHBOARD ----------------

        with dashboard_slot.container():

            st.subheader("📊 Job Detection Dashboard")

            col1, col2, col3, col4 = st.columns(4)

            total = st.session_state.total_jobs

            fake = st.session_state.fake_jobs

            genuine = st.session_state.genuine_jobs

            rate = (
                fake / total * 100
                if total > 0
                else 0
            )

            col1.metric("Total Jobs Analysed", total)

            col2.metric("Fake Jobs", fake)

            col3.metric("Genuine Jobs", genuine)

            col4.metric("Fake Job Rate", f"{rate:.2f}%")


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

        if all(
            col in data.columns
            for col in required_columns
        ):

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
                        sample[col]
                        .fillna("")
                        .astype(str)
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
                    "Potentially Fake"
                    if int(p) == 1
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

                result_columns = [
                    "title",
                    "Prediction",
                    "Warning Signs"
                ]

                st.dataframe(
                    sample[result_columns],
                    use_container_width=True
                )

                # ---------------- SUMMARY ----------------

                st.subheader("📈 Prediction Summary")

                counts = (
                    sample["Prediction"]
                    .value_counts()
                )

                fake_count = int(
                    (sample["Prediction"] == "Potentially Fake")
                    .sum()
                )

                genuine_count = int(
                    (sample["Prediction"] == "Likely Genuine")
                    .sum()
                )

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Jobs Analyzed",
                    len(sample)
                )

                col2.metric(
                    "Potentially Fake",
                    fake_count
                )

                col3.metric(
                    "Likely Genuine",
                    genuine_count
                )

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
                    "CSV predictions are based on the "
                    "trained Machine Learning model. "
                    "Warning signs are preliminary checks "
                    "and do not prove that a job is fraudulent."
                )

        else:

            st.warning(
                "CSV must contain these columns: "
                "title, company_profile, description "
                "and requirements."
            )

    except Exception as e:

        st.error(
            f"Could not process the dataset: {e}"
        )


# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "AI-Based Fake Job Detection System | "
    "Machine Learning Project | "
    "Always verify job offers independently."
)