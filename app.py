import os
import pickle

import pandas as pd
import streamlit as st

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(
    page_title="AI-Based Fake Job Detection System",
    page_icon="🛡️",
    layout="wide",
)

# ---------------- CSS DESIGN ----------------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #f5f7fb;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #172554;
    }

    .subtitle {
        color: #475569;
        font-size: 16px;
    }

    .result-fake {
        background: #fff1f2;
        border-left: 6px solid #e11d48;
        padding: 16px;
        border-radius: 10px;
        color: #881337;
    }

    .result-real {
        background: #ecfdf5;
        border-left: 6px solid #059669;
        padding: 16px;
        border-radius: 10px;
        color: #064e3b;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- LOAD MODEL FILES ----------------
@st.cache_resource
def load_artifacts():
    model_names = [
        "fake_job_model.pkl",
        "fake_job_model (1).pkl",
        "model.pkl",
    ]

    vectorizer_names = [
        "tfidf_vectorizer.pkl",
        "vectorizer.pkl",
        "tfidf.pkl",
    ]

    model = None
    vectorizer = None

    for filename in model_names:
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                model = pickle.load(file)
            break

    for filename in vectorizer_names:
        if os.path.exists(filename):
            with open(filename, "rb") as file:
                vectorizer = pickle.load(file)
            break

    return model, vectorizer


model, vectorizer = load_artifacts()

# ---------------- JOB TEXT COLUMNS ----------------
TEXT_COLUMNS = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "location",
    "department",
    "employment_type",
    "required_experience",
    "required_education",
    "industry",
    "function",
]


def clean_value(value):
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def combine_job_text(job):
    parts = []

    for column in TEXT_COLUMNS:
        if column in job:
            value = clean_value(job[column])

            if value:
                parts.append(value)

    return " ".join(parts)


# ---------------- WARNING REASONS ----------------
def get_warning_reasons(job):
    reasons = []
    text = combine_job_text(job).lower()

    if not clean_value(job.get("company_profile", "")):
        reasons.append("Company profile is missing")

    if not clean_value(job.get("requirements", "")):
        reasons.append("Job requirements are missing")

    if not clean_value(job.get("benefits", "")):
        reasons.append("Benefits information is missing")

    suspicious_phrases = [
        "registration fee",
        "training fee",
        "pay upfront",
        "guaranteed income",
        "earn money fast",
        "make money fast",
        "quick money",
        "wire transfer",
    ]

    matched_phrases = [
        phrase
        for phrase in suspicious_phrases
        if phrase in text
    ]

    if matched_phrases:
        reasons.append(
            "Suspicious wording: " + ", ".join(matched_phrases[:3])
        )

    logo = clean_value(job.get("has_company_logo", "")).lower()

    if logo in {"0", "false", "no"}:
        reasons.append("Company logo is not provided")

    if not reasons:
        reasons.append(
            "No obvious warning signs found by the basic checks"
        )

    return reasons


# ---------------- PREDICTION FUNCTION ----------------
def predict_job(job):
    if model is None or vectorizer is None:
        raise RuntimeError(
            "Model files are missing. Add the model and vectorizer "
            "files to the same repository as app.py."
        )

    text = combine_job_text(job)

    if not text.strip():
        raise ValueError("Please enter job details before checking.")

    features = vectorizer.transform([text])
    prediction = int(model.predict(features)[0])

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(features)[0]
        confidence = float(max(probabilities)) * 100

    if prediction == 1:
        label = "Potential Fake Job"
        reasons = get_warning_reasons(job)
    else:
        label = "Likely Genuine Job"
        reasons = [
            "The model did not classify this posting as fake.",
            "Verify the employer before accepting the job.",
        ]

    return label, confidence, reasons


# ---------------- HEADER ----------------
st.markdown(
    '<div class="main-title">'
    '🛡️ AI-Based Fake Job Detection System'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Detect suspicious job postings and review possible warning signs.'
    '</div>',
    unsafe_allow_html=True,
)

st.write("")

if model is None or vectorizer is None:
    st.error(
        "Model files not found. Keep app.py, your trained model file, "
        "and tfidf_vectorizer.pkl in the repository's main folder."
    )
else:
    st.success("AI model and TF-IDF vectorizer loaded successfully.")

# ---------------- SIDEBAR NAVIGATION ----------------
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Check a Job",
        "Bulk CSV Analysis",
        "About Project",
    ],
)

# ====================================================
# DASHBOARD
# ====================================================
if page == "Dashboard":
    st.subheader("📊 Job Detection Dashboard")

    previous_results = st.session_state.get("analysis_data")

    total_jobs = 0
    fake_jobs = 0
    genuine_jobs = 0

    if isinstance(previous_results, pd.DataFrame):
        total_jobs = len(previous_results)

        if "Prediction" in previous_results.columns:
            fake_jobs = int(
                (
                    previous_results["Prediction"]
                    == "Potential Fake Job"
                ).sum()
            )

            genuine_jobs = int(
                (
                    previous_results["Prediction"]
                    == "Likely Genuine Job"
                ).sum()
            )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Jobs Analysed", total_jobs)
    col2.metric("Potential Fake Jobs", fake_jobs)
    col3.metric("Likely Genuine Jobs", genuine_jobs)

    st.info(
        "Go to Bulk CSV Analysis to upload and analyse multiple job postings. "
        "Dashboard counts update after the analysis."
    )

    if isinstance(previous_results, pd.DataFrame):
        if not previous_results.empty:
            st.markdown("### Latest Analysis")
            st.dataframe(
                previous_results.head(10),
                use_container_width=True,
            )

            if "Prediction" in previous_results.columns:
                st.bar_chart(
                    previous_results["Prediction"].value_counts()
                )

# ====================================================
# CHECK A JOB
# ====================================================
elif page == "Check a Job":
    st.subheader("🔎 Check an Individual Job Posting")

    with st.form("single_job_form"):
        title = st.text_input("Job Title")
        company = st.text_input("Company Name")
        location = st.text_input("Location")

        company_profile = st.text_area("Company Profile")
        description = st.text_area("Job Description")
        requirements = st.text_area("Job Requirements")
        benefits = st.text_area("Job Benefits")

        submitted = st.form_submit_button("Check Job")

    if submitted:
        job = {
            "title": title,
            "company_profile": company_profile,
            "description": description,
            "requirements": requirements,
            "benefits": benefits,
            "location": location,
        }

        if company:
            job["company_profile"] = (
                company + " " + company_profile
            ).strip()

        try:
            label, confidence, reasons = predict_job(job)

            if label == "Potential Fake Job":
                st.markdown(
                    '<div class="result-fake">'
                    '<h3>⚠️ Potential Fake Job</h3>'
                    'This posting may be suspicious. Verify it carefully.'
                    '</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="result-real">'
                    '<h3>✅ Likely Genuine Job</h3>'
                    'The model did not flag this posting as fake. '
                    'This is not a guarantee.'
                    '</div>',
                    unsafe_allow_html=True,
                )

            if confidence is not None:
                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%",
                )

            st.markdown("### Warning Reasons / Review Notes")

            for reason in reasons:
                st.write(f"• {reason}")

            st.warning(
                "This prediction may be incorrect. Verify the company "
                "through official channels and never pay a fee to obtain a job."
            )

        except Exception as error:
            st.error(f"Could not analyse this job: {error}")

# ====================================================
# BULK CSV ANALYSIS
# ====================================================
elif page == "Bulk CSV Analysis":
    st.subheader("📁 Bulk Job Posting Analysis")

    st.write(
        "Upload a CSV dataset containing job postings. "
        "Columns such as title, description, requirements, and "
        "company_profile help the model analyse each job."
    )

    uploaded_file = st.file_uploader(
        "Upload Job Postings CSV File",
        type=["csv"],
    )

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)

            st.success("Dataset uploaded successfully.")

            st.write(
                f"Rows: {data.shape[0]} | Columns: {data.shape[1]}"
            )

            st.markdown("### Dataset Preview")

            st.dataframe(
                data.head(),
                use_container_width=True,
            )

            if data.empty:
                st.warning("The uploaded CSV file has no rows.")

            else:
                max_jobs = st.number_input(
                    "Number of jobs to analyse",
                    min_value=1,
                    max_value=len(data),
                    value=min(100, len(data)),
                    step=1,
                )

                if st.button("Analyse Jobs"):
                    if model is None or vectorizer is None:
                        st.error(
                            "Model files are missing. Add your model "
                            "and vectorizer files to GitHub."
                        )

                    else:
                        total_to_analyze = min(
                            int(max_jobs),
                            len(data),
                        )

                        results = []

                        progress = st.progress(0)
                        status = st.empty()

                        for index, (_, row) in enumerate(
                            data.head(total_to_analyze).iterrows()
                        ):
                            job = row.to_dict()
                            result_row = row.to_dict()

                            try:
                                label, confidence, reasons = predict_job(
                                    job
                                )

                                result_row["Prediction"] = label

                                if confidence is not None:
                                    result_row["Confidence_Percent"] = (
                                        round(confidence, 2)
                                    )
                                else:
                                    result_row["Confidence_Percent"] = "N/A"

                                result_row["Warning_Reasons"] = (
                                    " | ".join(reasons)
                                )

                            except Exception as error:
                                result_row["Prediction"] = "Analysis Error"
                                result_row["Confidence_Percent"] = "N/A"
                                result_row["Warning_Reasons"] = str(error)

                            results.append(result_row)

                            progress.progress(
                                (index + 1) / total_to_analyze
                            )

                            status.write(
                                f"Analysed {index + 1} "
                                f"of {total_to_analyze} jobs..."
                            )

                        results_df = pd.DataFrame(results)

                        st.session_state["analysis_data"] = results_df

                        st.success("Analysis completed!")

                        st.markdown("### Analysis Results")

                        st.dataframe(
                            results_df,
                            use_container_width=True,
                        )

                        fake_count = int(
                            (
                                results_df["Prediction"]
                                == "Potential Fake Job"
                            ).sum()
                        )

                        genuine_count = int(
                            (
                                results_df["Prediction"]
                                == "Likely Genuine Job"
                            ).sum()
                        )

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Jobs Analysed",
                            len(results_df),
                        )

                        col2.metric(
                            "Potential Fake Jobs",
                            fake_count,
                        )

                        col3.metric(
                            "Likely Genuine Jobs",
                            genuine_count,
                        )

                        st.markdown("### Prediction Chart")

                        st.bar_chart(
                            results_df["Prediction"].value_counts()
                        )

                        csv_bytes = results_df.to_csv(
                            index=False
                        ).encode("utf-8")

                        st.download_button(
                            label="Download Analysis Results (CSV)",
                            data=csv_bytes,
                            file_name="fake_job_analysis_results.csv",
                            mime="text/csv",
                        )

        except Exception as error:
            st.error(
                f"Could not read or analyse the CSV file: {error}"
            )

# ====================================================
# ABOUT PROJECT
# ====================================================
elif page == "About Project":
    st.subheader("ℹ️ About This Project")

    st.write(
        "The AI-Based Fake Job Detection System uses a trained "
        "machine-learning model and a TF-IDF text vectorizer to "
        "classify job postings as potentially fake or likely genuine."
    )

    st.markdown(
        "**Dataset:** Real or Fake Job Posting Dataset from Kaggle."
    )

    st.markdown("### Main Features")

    st.write("• Individual job posting prediction")
    st.write("• Bulk CSV job analysis")
    st.write("• Possible warning reasons")
    st.write("• Model confidence when available")
    st.write("• Dashboard and prediction chart")
    st.write("• Downloadable CSV results")

    st.warning(
        "The model provides an estimate, not proof that a job is fake "
        "or genuine. Always verify the employer independently."
    )