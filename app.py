import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# -----------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------
st.set_page_config(
    page_title="Student Success Dashboard – Silverleaf University",
    layout="wide",
    page_icon="🎓"
)

# -----------------------------------------------------------
# GLOBAL STYLE (CSS)
# -----------------------------------------------------------
st.markdown(
    """
<style>
/* BASE APP */
html, body, .stApp {
    font-family: "Inter", system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background-color: #f5f7fb;
}

/* TITLES */
h1, h2, h3, h4, h5 {
    color: #14213d;
}

/* KPI CARDS */
.metric-card {
    padding: 16px;
    border-radius: 14px;
    text-align: center;
    color: #ffffff;
    font-weight: 600;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.18);
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.metric-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(15, 23, 42, 0.22);
}
.metric-label {
    font-size: 12px;
    opacity: 0.9;
    letter-spacing: 0.03em;
    text-transform: uppercase;
}
.metric-value {
    font-size: 22px;
    margin-top: 4px;
}

/* INSIGHT CARDS */
.insight-card {
    border-radius: 10px;
    padding: 12px 12px 10px 12px;
    margin-bottom: 10px;
    background: #ffffff;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.12);
    border-left-width: 5px;
    border-left-style: solid;
    animation: fadeIn 0.25s ease-in-out;
}
.insight-card.green { border-left-color: #16a34a; }
.insight-card.yellow { border-left-color: #f59e0b; }
.insight-card.blue   { border-left-color: #2563eb; }
.insight-card.gray   { border-left-color: #6b7280; }
.insight-title { font-weight: 600; font-size: 12px; margin-bottom: 3px; }
.insight-body  { font-size: 12px; }

/* PANEL HEADER */
.panel-header {
    font-weight: 600;
    font-size: 13px;
    margin-bottom: 6px;
    color: #111827;
}

/* ACADEMIC ENGAGEMENT METRICS */
.metric-box {
    background: #ffffff;
    border-radius: 14px;
    padding: 16px 18px;
    width: 100% !important;
    display: block !important;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.1);
    margin-top: 8px;
}
.metric-row {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #f1f5f9;
}
.metric-row:last-child {
    border-bottom: none;
}
.metric-row label {
    font-weight: 500;
    color: #111827;
    font-size: 13px;
}
.metric-row span {
    font-weight: 600;
    color: #1d4ed8;
    font-size: 13px;
}

/* DROPDOWN SCROLLER */
div[data-baseweb="select"] > div[role="listbox"] {
    max-height: 220px;
    overflow-y: auto;
}

/* ANALYZE BUTTON CARD */
.analyze-card {
    background: linear-gradient(135deg, #1d4ed8 0%, #06b6d4 100%);
    color: #e5e7eb;
    border-radius: 14px;
    padding: 14px 16px;
    box-shadow: 0 5px 18px rgba(37, 99, 235, 0.35);
}

/* FOOTER */
.footer-container {
    width: 100%;
    text-align: center;
    padding: 30px 0 18px 0;
    margin-top: 40px;
    color: #6b7280;
    font-size: 13px;
}
.footer-note {
    font-weight: 500;
}
.footer-copy {
    font-size: 12px;
    opacity: 0.9;
}

/* SIMPLE FADE ANIMATION */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------
# CONSTANTS
# -----------------------------------------------------------
PROGRAM_COURSE_MAP = {
    "Business Analytics": [
        "AI Basics",
        "Statistics",
        "Programming",
        "Communication Skills",
        "Predictive Modeling",
    ],
    "Data Science": [
        "Python Programming",
        "Machine Learning",
        "Data Wrangling",
        "Deep Learning",
        "Visualization",
    ],
    "Computer Science": [
        "Algorithms",
        "Operating Systems",
        "Networks",
        "Java Programming",
        "Databases",
        "Advanced Networking",
    ],
    "Engineering": [
        "Physics",
        "Calculus",
        "Materials Science",
        "CAD",
        "Signals",
    ],
}

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
@st.cache_data
def load_data():
    """
    Load all CSVs and build a master table.
    Adjust the paths if needed.
    """
    base = Path("C:/ASSIGNMENTS/MRP Invincible/data")  # adjust if you move the project

    profile = pd.read_csv(r"student_profile.csv")
    perf = pd.read_csv(r"student_performance.csv")
    enroll = pd.read_csv(r"course_enrollment.csv")
    engage = pd.read_csv(r"engagement_log.csv")
    insights = pd.read_csv(r"ai_insights.csv")

    # Normalize Student_ID type
    for df in [profile, perf, enroll, engage, insights]:
        if "Student_ID" in df.columns:
            df["Student_ID"] = df["Student_ID"].astype(str)

    # Build master table (student + performance + ai metadata)
    master = (
        profile.merge(perf, on="Student_ID", how="inner")
        .merge(insights[["Student_ID", "Risk_Level", "Message", "Semester"]],
               on="Student_ID", how="left")
    )

    return profile, perf, enroll, engage, insights, master


profile_df, perf_df, enroll_df, engage_df, insights_df, master_df = load_data()

# Try to load trained risk model (optional)
model_path = Path("risk_model.pkl")
risk_model = joblib.load(model_path) if model_path.exists() else None

# -----------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"

default_state = {
    "student_id": None,
    "student_name": None,
    "program": None,
    "year": None,
    "semester": None,
    "course_grades": {},
    "study_hours": 10,
    "attendance": 80,
    "motivation": 6,
}
for k, v in default_state.items():
    st.session_state.setdefault(k, v)

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def compute_kpis(row):
    """Return Attendance, ExamScore, StudyHours, Motivation safely."""
    if row is None or row.empty:
        return 0, 0, 0, 0
    r = row.iloc[0]
    att = float(r.get("Attendance", 0))
    exam = float(r.get("ExamScore", 0))
    hours = float(r.get("StudyHours", 0))
    mot = float(r.get("Motivation", 0))
    return round(att, 1), round(exam, 1), round(hours, 1), round(mot, 1)


def build_weekly_hours_trend(student_row, cohort_df, current_hours):
    """
    Simulate a weekly trend of study hours for the student vs cohort.
    The trend responds to the Study Hours slider.
    """
    if student_row.empty:
        return pd.DataFrame()

    weeks = ["Week 2", "Week 4", "Week 8", "Week 14"]

    # Student hours: ramp up to current slider value
    student_vals = [
        current_hours * 0.4,
        current_hours * 0.7,
        current_hours * 0.9,
        current_hours,
    ]

    # Cohort hours based on average
    cohort_hours = float(cohort_df["StudyHours"].mean()) if "StudyHours" in cohort_df.columns else 8.0
    cohort_vals = [
        cohort_hours * 0.4,
        cohort_hours * 0.7,
        cohort_hours * 0.9,
        cohort_hours,
    ]

    df = pd.DataFrame(
        {
            "Week": weeks,
            "Student Study Hours": student_vals,
            "Cohort Study Hours": cohort_vals,
        }
    )
    return df


def build_resource_distribution(student_row, study_hours):
    """Estimate how the student splits study time across resources."""
    if student_row.empty:
        return pd.DataFrame()

    r = student_row.iloc[0]
    online = float(r.get("EduTech", 0))  # usage of edutech tools
    tutoring = float(r.get("Extracurricular", 0))  # extra sessions / peer support
    # assign remaining time as in-person classes / self-study
    in_person = max(study_hours - online - tutoring, 0)

    return pd.DataFrame(
        {
            "Resource": ["Online Learning", "In-Person Classes", "Tutoring Sessions"],
            "Hours": [online, in_person, tutoring],
        }
    )


def build_engagement_metrics(student_id):
    """Create friendly engagement metrics for the right-hand card."""
    row = perf_df[perf_df["Student_ID"] == student_id]
    if row.empty:
        row = perf_df.sample(1, random_state=0)
    r = row.iloc[0]

    ai_tutor = int(round(r.get("OnlineCourses", 0) / 2))
    study_group = int(round(r.get("Extracurricular", 0)))
    logins = int(round(r.get("Resources", 0) * 4))
    downloads = int(round(r.get("EduTech", 0) * 8))
    assign_comp = float(r.get("AssignmentCompletion", 0))
    stress_val = float(r.get("StressLevel", 0))
    forum_posts = int(round(r.get("Discussions", 0)))

    if stress_val >= 7:
        stress_label = "High"
    elif stress_val >= 4:
        stress_label = "Medium"
    else:
        stress_label = "Low"

    if assign_comp >= 75 and stress_val <= 6:
        status = "On Track"
    elif assign_comp >= 50:
        status = "Needs Attention"
    else:
        status = "At Risk"

    return {
        "AI Tutor Sessions": ai_tutor,
        "Study Group Sessions": study_group,
        "Dashboard Logins": logins,
        "Resource Downloads": downloads,
        "Assignment Completion": f"{assign_comp:.1f}%",
        "Stress Level": stress_label,
        "Forum Participation": f"{forum_posts} posts",
        "Overall Progress": status,
    }

def rule_based_risk(student_row):
    """Improved rule-based risk using actual signals from updated student data."""
    if student_row.empty:
        return "Unknown"

    r = student_row.iloc[0]

    attendance = float(r.get("Attendance", 0))
    exam = float(r.get("ExamScore", 0))
    assignment = float(r.get("AssignmentCompletion", 0))
    stress = float(r.get("StressLevel", 0))

    # HIGH RISK
    if attendance < 60 or exam < 50 or assignment < 50 or stress >= 8:
        return "High"

    # MEDIUM RISK
    if attendance < 75 or exam < 65 or assignment < 65 or stress >= 6:
        return "Medium"

    # LOW RISK
    return "Low"

def predict_risk(student_row):
    """Hybrid risk engine combining rule-based checks with ML predictions."""
    if student_row.empty:
        return "Unknown"

    # STEP 1 — Always start with rule-based risk
    rule_risk = rule_based_risk(student_row)

    # LOW cannot be overridden (protects from ML inaccuracies)
    if rule_risk == "Low":
        return "Low"

    # If no ML model exists, use rule-based
    if risk_model is None:
        return rule_risk

    # Select valid model features
    candidate_cols = [
        "StudyHours", "Attendance", "Resources", "Extracurricular",
        "Motivation", "Internet", "Gender", "Age", "LearningStyle",
        "OnlineCourses", "Discussions", "AssignmentCompletion",
        "ExamScore", "EduTech", "StressLevel", "FinalGrade"
    ]
    cols = [c for c in candidate_cols if c in student_row.columns]

    if not cols:
        return rule_risk

    X = student_row[cols].astype(float)

    # STEP 2 — ML prediction
    try:
        ml_pred = risk_model.predict(X)[0]
    except:
        return rule_risk

    # STEP 3 — hybrid arbitration
    # If rule = Medium but ML = High → Medium (avoid ML over-harshness)
    if rule_risk == "Medium" and ml_pred == "High":
        return "Medium"

    return ml_pred


def local_ai_agent(student_row, risk_level):
    """
    Generate plain-language messages based on local data only.
    No external AI calls.
    """
    if student_row.empty:
        msg = "We could not find your record in the system. Please contact your program office."
        return msg, msg, msg, msg

    r = student_row.iloc[0]
    att = float(r.get("Attendance", 0))
    study = float(r.get("StudyHours", 0))
    stress = float(r.get("StressLevel", 0))
    exams = float(r.get("ExamScore", 0))
    assign_comp = float(r.get("AssignmentCompletion", 0))

    # Positive note
    if exams >= 80 and assign_comp >= 85:
        positive = "You are doing very well. Your exam scores and assignment completion are strong."
    elif exams >= 65:
        positive = "You are keeping up with most of the material. There is a good foundation to build on."
    else:
        positive = "You have made a start. Checking this dashboard is a good step toward improving your progress."

    # Improvement focus
    improvements = []
    if att < 70:
        improvements.append("Try to raise your attendance closer to at least 80%.")
    if study < 10:
        improvements.append("Aim for at least 10–12 focused study hours per week.")
    if assign_comp < 70:
        improvements.append("Try to submit all assignments on time and avoid missing deadlines.")

    if improvements:
        improvement_msg = " ".join(improvements)
    else:
        improvement_msg = "Your current learning habits look stable. Keep refining what already works for you."

    # Stress tip
    if stress >= 8:
        tip = "Your stress level looks high. Please consider talking with an advisor and taking regular breaks."
    elif stress >= 5:
        tip = "Your stress is moderate. Balance deep study sessions with short walks, breaks, and enough sleep."
    else:
        tip = "Your stress level appears manageable. Keep protecting your downtime and healthy routines."

    # Summary
    summary = (
        f"Right now your risk level is **{risk_level}**. "
        f"Attendance is **{att:.0f}%**, weekly study hours are about **{study:.1f}**, "
        f"your main exam score is **{exams:.0f}**, and assignment completion is around **{assign_comp:.0f}%**. "
        "These metrics drive the insights shown on your dashboard."
    )

    return positive, improvement_msg, tip, summary


# -----------------------------------------------------------
# HOME / LANDING PAGE
# -----------------------------------------------------------
def show_home():
    st.title("Silverleaf University – Student Success Dashboard")

    st.write("""
    Welcome to the **Student Success Dashboard**!  
    This tool is designed to help you understand how you're performing academically, how your study habits are shaping your results, and what steps you can take to stay on track throughout the semester.

    ### 📌 What this dashboard does
    - Helps you **track your course performance** using your most recent grades  
    - Lets you **review your study habits**, attendance, and overall motivation  
    - Gives you **AI-generated insights** based on your data  
    - Provides an easy way to understand where you’re doing well and where you might need support  

    ### 🧭 How to get started
    1. **Find your name** using the search box.  
       Once selected, your program, year, and semester will automatically load based on our records.
    2. **Review your details**  
       Program is fixed based on your academic path, but you can adjust semester and year if you want to view previous information.
    3. **Enter your latest course grades**  
       Your courses will appear based on your program. Just enter your most recent grades for each course.
    4. **Complete the quick study survey**  
       Tell us about your weekly study hours, attendance for this semester, and your motivation level.
    5. When you're ready, click **Go to Dashboard**.  
       We'll generate your personalized dashboard with performance charts, study insights, and academic engagement metrics.

    ### 🎯 Why this helps
    This dashboard brings all your key information into one place so you can:
    - Understand how your habits influence your grades  
    - Compare your performance trends with expected progress  
    - Get personalized suggestions powered by our AI model  
    - Take early action before falling behind  

    You're just a few steps away from seeing your academic journey more clearly—let’s begin!
    """)

    # -----------------------------
    # STEP 1 – SELECT NAME
    # -----------------------------
    st.subheader("Step 1: Find your name")

    selected_name = st.selectbox(
        "Search or select your name:",
        options=sorted(profile_df["Name"].unique()),
    )

    # Fetch the student's record
    prof_row = profile_df[profile_df["Name"] == selected_name].iloc[0]
    student_id = prof_row["Student_ID"]
    default_program = prof_row["Program"]
    default_year = prof_row["Year"]

    # Semester from insights (latest if multiple), otherwise Fall 2024
    user_insights = insights_df[insights_df["Student_ID"] == str(student_id)]
    if not user_insights.empty:
        default_semester = user_insights.iloc[0]["Semester"]
    else:
        default_semester = "Fall 2024"

    # Store core identity in session
    st.session_state["student_name"] = selected_name
    st.session_state["student_id"] = str(student_id)
    st.session_state["program"] = default_program

    # -----------------------------
    # STEP 2 – PROGRAM DETAILS
    # -----------------------------
    st.subheader("Step 2: Confirm your program details")

    col_sem, col_prog, col_year = st.columns([2, 3, 1.5])

    with col_sem:
        st.caption("Semester")
        semester_input = st.selectbox(
            "",
            options=["Fall 2023", "Spring 2024", "Fall 2024", "Spring 2025"],
            index=["Fall 2023", "Spring 2024", "Fall 2024", "Spring 2025"].index(
                default_semester
            )
            if default_semester in ["Fall 2023", "Spring 2024", "Fall 2024", "Spring 2025"]
            else 2,
            key="home_semester",
        )

    with col_prog:
        st.caption("Program (fixed from records)")
        st.selectbox(
            "",
            options=[default_program],
            disabled=True,
            key="home_program",
        )

    with col_year:
        st.caption("Year")
        year_input = st.selectbox(
            "",
            options=["1st", "2nd", "3rd", "4th"],
            index=["1st", "2nd", "3rd", "4th"].index(default_year),
            key="home_year",
        )

    # Save year/semester to session
    st.session_state["year"] = year_input
    st.session_state["semester"] = semester_input

    # -----------------------------
    # STEP 3 – COURSE GRADES
    # -----------------------------
    st.subheader("Step 3: Enter your latest course grades")

    program_courses = PROGRAM_COURSE_MAP.get(default_program, [])
    course_grades = {}

    for course in program_courses:
        course_grades[course] = st.slider(
            f"{course} grade (0–100)",
            min_value=0,
            max_value=100,
            value=80,
        )

    st.session_state["course_grades"] = course_grades

    # -----------------------------
    # STEP 4 – STUDY SURVEY
    # -----------------------------
    st.subheader("Step 4: Quick study survey")

    col_hours, col_att, col_mot = st.columns(3)

    with col_hours:
        study_hours = st.slider(
            "Study hours per week",
            0,
            60,
            st.session_state.get("study_hours", 10),
        )

    with col_att:
        attendance = st.slider(
            "Attendance this semester (%)",
            0,
            100,
            st.session_state.get("attendance", 80),
        )

    with col_mot:
        motivation = st.slider(
            "Motivation level (0–10)",
            0,
            10,
            st.session_state.get("motivation", 6),
        )

    st.session_state["study_hours"] = study_hours
    st.session_state["attendance"] = attendance
    st.session_state["motivation"] = motivation

    st.markdown("---")

    if st.button("Go to Dashboard", use_container_width=True):
        st.session_state["page"] = "dashboard"
        st.rerun()


# -----------------------------------------------------------
# DASHBOARD PAGE
# -----------------------------------------------------------
def show_dashboard():
    # Guard: if no student selected, go back home
    if not st.session_state.get("student_id"):
        st.session_state["page"] = "home"
        st.rerun()

    student_id = st.session_state["student_id"]
    student_name = st.session_state["student_name"]
    semester_state = st.session_state["semester"]
    program_state = st.session_state["program"]
    year_state = st.session_state["year"]
    latest_grades = st.session_state.get("course_grades", {})

    # NEW FIX — add this:
    study_hours = st.session_state["study_hours"]
    attendance = st.session_state["attendance"]
    motivation = st.session_state["motivation"]

    # HEADER
    top_left, top_right = st.columns([3.5, 2.0])
    with top_left:
        st.subheader("Student Success Dashboard – Silverleaf University")
        st.caption("Track your progress. Discover insights. Improve outcomes.")

    with top_right:
        st.write(f"**Logged in as:** {student_name}")
        if st.button("🔙 Back to Home", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()

    st.markdown("---")

    # LAYOUT: Filters / Main / Right
    filters_col, main_col, right_col = st.columns([1.0, 2.5, 1.5])

    # -----------------------------
    # FILTERS (LEFT COLUMN)
    # -----------------------------
    with filters_col:
        st.markdown("### Filters")

        # --- Semester (editable) ---
        semesters = ["Fall 2023", "Spring 2024", "Fall 2024", "Spring 2025"]
        semester = st.selectbox(
            "Semester",
            options=semesters,
            index=semesters.index(st.session_state["semester"])
        )

        st.session_state["semester"] = semester

        # --- Program (FROZEN) ---
        st.selectbox(
            "Program",
            options=[st.session_state["program"]],
            disabled=True
        )

        # --- Year (editable) ---
        years = ["1st", "2nd", "3rd", "4th"]
        year = st.selectbox(
            "Year",
            options=years,
            index=years.index(st.session_state["year"])
        )
        st.session_state["year"] = year

        # --- Course Filter ---
        courses_all = ["All Courses"] + PROGRAM_COURSE_MAP[st.session_state["program"]]
        course_filter = st.selectbox("Course:", options=courses_all)

        # --- Remove Risk Level + Attendance Range (NO LONGER NEEDED) ---

        st.markdown("### Adjust Inputs")

        # Study hours (dashboard + home synced)
        new_study = st.slider(
            "Study Hours / Week",
            0, 60,
            value=st.session_state["study_hours"]
        )
        st.session_state["study_hours"] = new_study

        # Attendance sync
        new_attendance = st.slider(
            "Attendance (%)",
            0, 100,
            value=st.session_state["attendance"]
        )
        st.session_state["attendance"] = new_attendance

        # Motivation sync
        new_motivation = st.slider(
            "Motivation (0–10)",
            0, 10,
            value=st.session_state["motivation"]
        )
        st.session_state["motivation"] = new_motivation

    # -----------------------------
    # BUILD COHORT BASED ON FILTERS
    # -----------------------------
    cohort = master_df.copy()

    # Filter by current semester
    cohort = cohort[cohort["Semester"] == semester]

    # Filter by student's program (always frozen)
    cohort = cohort[cohort["Program"] == program_state]

    # Filter by year
    cohort = cohort[cohort["Year"] == year_state]

    # If selecting a specific course, pick students enrolled in that course
    if course_filter != "All Courses":
        stu_ids_for_course = enroll_df[enroll_df["Course_Name"] == course_filter]["Student_ID"].unique()
        cohort = cohort[cohort["Student_ID"].isin(stu_ids_for_course)]

    # Ensure we do not break visuals
    if cohort.empty:
        cohort = master_df.copy()

    # -----------------------------
    # BUILD STUDENT ROW (ADJUSTED)
    # -----------------------------
    student_row = master_df[master_df["Student_ID"] == str(student_id)]
    if student_row.empty:
        student_row = cohort[cohort["Student_ID"] == str(student_id)]

    student_row_adj = student_row.copy()
    if not student_row_adj.empty:
        idx = student_row_adj.index[0]
        student_row_adj.loc[idx, "StudyHours"] = study_hours
        student_row_adj.loc[idx, "Attendance"] = attendance
        student_row_adj.loc[idx, "Motivation"] = motivation

        if latest_grades:
            avg_new_exam = float(np.mean(list(latest_grades.values())))
            student_row_adj.loc[idx, "ExamScore"] = avg_new_exam
            student_row_adj.loc[idx, "FinalGrade"] = avg_new_exam

    # -----------------------------
    # KPI ROW (MAIN COLUMN TOP)
    # -----------------------------
    with main_col:
        k1, k2, k3, k4 = st.columns(4)
        k_att, k_exam, k_hours, k_mot = compute_kpis(student_row_adj)

        with k1:
            st.markdown(
                f"""
                <div class="metric-card" style="background: linear-gradient(135deg,#fb7185,#f97316);">
                    <div class="metric-label">Attendance</div>
                    <div class="metric-value">{k_att:.1f}%</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="metric-card" style="background: linear-gradient(135deg,#22c55e,#16a34a);">
                    <div class="metric-label">Avg Exam Score</div>
                    <div class="metric-value">{k_exam:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="metric-card" style="background: linear-gradient(135deg,#f59e0b,#eab308);">
                    <div class="metric-label">Study Hours / Week</div>
                    <div class="metric-value">{k_hours:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k4:
            st.markdown(
                f"""
                <div class="metric-card" style="background: linear-gradient(135deg,#3b82f6,#1d4ed8);">
                    <div class="metric-label">Motivation (0–10)</div>
                    <div class="metric-value">{k_mot:.1f}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # -----------------------------
    # MAIN VISUALS (COURSE, TREND, RESOURCES, ENGAGEMENT)
    # -----------------------------
    with main_col:
        
        # COURSE PERFORMANCE OVERVIEW
        hist = enroll_df[enroll_df["Student_ID"] == student_id][["Course_Name", "Grade"]]
        hist = hist.groupby("Course_Name", as_index=False)["Grade"].mean()
        hist.rename(columns={"Grade": "Previous_Grade"}, inplace=True)

        latest_list = [{"Course_Name": c, "Current_Grade": g}
                       for c, g in latest_grades.items()]
        latest_df = pd.DataFrame(latest_list)

        if hist.empty and latest_df.empty:
            course_perf = pd.DataFrame(columns=["Course_Name", "Previous_Grade", "Current_Grade"])
        else:
            course_perf = pd.merge(hist, latest_df, on="Course_Name", how="outer")

        st.markdown("#### Course Performance Overview")

        # Apply course filter
        if course_filter != "All Courses":
            course_perf = course_perf[course_perf["Course_Name"] == course_filter]

        if not course_perf.empty:
            course_perf.fillna(0, inplace=True)
            fig_course = px.bar(
                course_perf,
                x="Course_Name",
                y=["Previous_Grade", "Current_Grade"],
                barmode="group",
            )
            fig_course.update_layout(
                showlegend=False,
                height=260,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis_title="Grade"
            )
            st.plotly_chart(fig_course, use_container_width=True)
        else:
            st.info("No course information available.")

        # WEEKLY STUDY HOURS TREND
        st.markdown("#### Weekly Study Hours & Performance Correlation")

        weekly_df = build_weekly_hours_trend(student_row_adj, cohort, study_hours)

        if not weekly_df.empty:
            fig_week = px.line(
                weekly_df,
                x="Week",
                y=["Student Study Hours", "Cohort Study Hours"],
                markers=True,
            )
            fig_week.update_layout(
                height=260,
                margin=dict(l=20, r=20, t=30, b=10),
                yaxis_title="Study Hours",
            )
            st.plotly_chart(fig_week, use_container_width=True)
        else:
            st.info("Not enough information to show a weekly trend.")

        # STUDY RESOURCES + ENGAGEMENT
        col_res, col_eng = st.columns(2)

        with col_res:
            st.markdown("#### Study Resources Distribution")
            res_df = build_resource_distribution(student_row_adj, study_hours)
            if not res_df.empty:
                fig_res = px.pie(
                    res_df,
                    names="Resource",
                    values="Hours",
                    hole=0.55,
                )
                fig_res.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=30, b=10),
                )
                st.plotly_chart(fig_res, use_container_width=True)
            else:
                st.info("No resource usage information available.")

        with col_eng:
            st.markdown("#### Academic Engagement Metrics")

            eng = build_engagement_metrics(str(student_id))

                        # Render each metric in a clean professional row
            for label in eng:
                st.markdown(
                    f"""
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        padding:10px 0;
                        border-bottom:1px solid #eee;
                        font-size:15px;
                    ">
                        <span style="font-weight:600; color:#222;">{label}</span>
                        <span style="font-weight:600; color:#1b4ed8;">{eng[label]}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # -----------------------------
    # RIGHT COLUMN – AI INSIGHTS
    # -----------------------------
    with right_col:
        st.markdown(
            """
            <div class="analyze-card">
                <div style="font-weight:600; font-size:13px; margin-bottom:6px;">
                    Analyze My Performance
                </div>
                <div style="font-size:11px; line-height:1.5;">
                    Click the button below to generate personalized academic insights
                    based on your latest grades and survey responses. All insights are
                    generated locally from your data.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        analyze_clicked = st.button("🔍 Analyze My Performance", use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown("""
            <div style="
                font-size:18px;
                font-weight:700;
                margin-top:25px;
                margin-bottom:10px;
                color:#1a237e;
            ">
                🎓 AI-Generated Academic Insights
            </div>
        """, unsafe_allow_html=True)

        if not analyze_clicked:
            st.info(
                "No analysis yet. Click **Analyze My Performance** to see your personalized insights."
            )
        else:
            risk_level = predict_risk(student_row_adj)
            pos_msg, improv_msg, tip_msg, summary_msg = local_ai_agent(
                student_row_adj, risk_level
            )

            # Color card for risk level
            if risk_level == "High":
                status_color = "yellow"
                status_text = "You may be at higher risk this term. Please review the suggestions carefully."
            elif risk_level == "Medium":
                status_color = "blue"
                status_text = "You are in the middle range. With a few adjustments, you can move to a safer zone."
            elif risk_level == "Low":
                status_color = "green"
                status_text = "You appear to be on track. Keep up your current momentum."
            else:
                status_color = "gray"
                status_text = "Risk level is unknown. Some data may be missing."

            st.markdown(
                f"""
                <div class="insight-card {status_color}">
                    <div class="insight-title">📌 Risk Overview – {risk_level}</div>
                    <div class="insight-body">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="insight-card green">
                    <div class="insight-title">✅ Positive Reinforcement</div>
                    <div class="insight-body">{pos_msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="insight-card yellow">
                    <div class="insight-title">⚠️ Improvement Suggestion</div>
                    <div class="insight-body">{improv_msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="insight-card blue">
                    <div class="insight-title">💡 Personalized Tip</div>
                    <div class="insight-body">{tip_msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(
                f"""
                <div class="insight-card gray">
                    <div class="insight-title">📊 Overall Summary</div>
                    <div class="insight-body">{summary_msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # FOOTER
    st.markdown(
        """
        <div class="footer-container">
            <div class="footer-note">
                This dashboard is a tool for student academic success insights.
            </div>
            <div class="footer-copy">
                © 2025 Silverleaf University · All Rights Reserved · Student Success Dashboard
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------
# ROUTER
# -----------------------------------------------------------
if st.session_state["page"] == "home":
    show_home()
else:
    show_dashboard()
