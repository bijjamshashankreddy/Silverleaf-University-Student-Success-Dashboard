import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from pathlib import Path

# -----------------------------------------------------------
# PAGE CONFIG + BASIC STYLING
# -----------------------------------------------------------
st.set_page_config(
    page_title="Student Success Dashboard – Silverleaf University",
    layout="wide",
)

st.markdown("""
<style>

/* GENERAL */
body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f5f7fb;
}

/* KPI CARDS */
.metric-card {
    padding: 16px;
    border-radius: 12px;
    text-align: center;
    color: white;
    font-weight: 600;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
}
.metric-label {
    font-size: 12px;
    opacity: 0.9;
}
.metric-value {
    font-size: 22px;
    margin-top: 4px;
}

/* INSIGHT CARDS */
.insight-card {
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 10px;
    background: #ffffff;
    box-shadow: 0 2px 5px rgba(0,0,0,0.07);
    border-left-width: 5px;
    border-left-style: solid;
}
.insight-card.yellow { border-left-color: #ffb400; }
.insight-card.blue   { border-left-color: #2196f3; }
.insight-title { font-weight: 600; font-size: 12px; margin-bottom: 3px; }
.insight-body  { font-size: 12px; }

/* ACADEMIC ENGAGEMENT METRICS – CLEAN + PRO */
.metric-wrapper {
    width: 100%;
    display: block !important;
}

.metric-box {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    width: 100% !important;
    display: block !important;
    box-shadow: 0 3px 10px rgba(0,0,0,0.08);
    margin-top: 10px;
    position: relative;
}

.metric-row {
    width: 100%;
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid #f0f0f0;
}

.metric-row:last-child {
    border-bottom: none;
}

.metric-row label {
    font-weight: 600;
    color: #222;
}

.metric-row span {
    font-weight: 600;
    color: #1b4ed8;
}

/* SCROLLBAR FOR DROPDOWN */
div[data-baseweb="select"] > div[role="listbox"] {
    max-height: 220px;
    overflow-y: auto;
}

/* FOOTER NOTES */
.footer-container {
    width: 100%;
    text-align: center;
    padding: 25px 0 15px 0;
    margin-top: 40px;
    color: #666;
    font-size: 13px;
}

.footer-note {
    font-weight: 500;
}

.footer-copy {
    font-size: 12px;
    opacity: 0.8;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------
@st.cache_data
def load_data():
    profile = pd.read_csv(r"C:\ASSIGNMENTS\MRP Invincible\data\student_profile.csv")
    perf = pd.read_csv(r"C:\ASSIGNMENTS\MRP Invincible\data\student_performance.csv")
    enroll = pd.read_csv(r"C:\ASSIGNMENTS\MRP Invincible\data\course_enrollment.csv")
    engage = pd.read_csv(r"C:\ASSIGNMENTS\MRP Invincible\data\engagement_log.csv")
    insights = pd.read_csv(r"C:\ASSIGNMENTS\MRP Invincible\data\ai_insights.csv")

    for df in [profile, perf, enroll, engage, insights]:
        if "Student_ID" in df.columns:
            df["Student_ID"] = df["Student_ID"].astype(str)

    master = (
        profile
        .merge(perf, on="Student_ID", how="inner")
        .merge(insights[["Student_ID", "Risk_Level", "Message", "Semester"]],
               on="Student_ID", how="left")
    )

    # attach Program to enrollment so we know courses by program
    enroll_prog = enroll.merge(
        profile[["Student_ID", "Program"]],
        on="Student_ID",
        how="left"
    )

    return profile, perf, enroll, enroll_prog, engage, insights, master


profile_df, perf_df, enroll_df, enroll_prog_df, engage_df, insights_df, master_df = load_data()

model_path = Path("risk_model.pkl")
risk_model = joblib.load(model_path) if model_path.exists() else None

# -----------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "home"

# these will be set on home page
defaults = {
    "selected_student_id": None,
    "selected_student_name": None,
    "selected_program": None,
    "selected_year": None,
    "selected_semester": None,
    "latest_grades": {},
    "survey_study_hours": None,
    "survey_motivation": None,
    "survey_attendance": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# -----------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------
def get_courses_for_program(program, max_courses=4):
    df = enroll_prog_df[enroll_prog_df["Program"] == program]
    if df.empty:
        # fallback: most common courses overall
        return (
            enroll_df["Course_Name"]
            .value_counts()
            .index
            .tolist()[:max_courses]
        )
    return (
        df["Course_Name"]
        .value_counts()
        .index
        .tolist()[:max_courses]
    )


def compute_kpis(student_row_adj):
    if student_row_adj is None or student_row_adj.empty:
        return 0, 0, 0, 0.0
    att = float(student_row_adj.get("Attendance", 0))
    exam = float(student_row_adj.get("ExamScore", 0))
    hours = float(student_row_adj.get("StudyHours", 0))
    mot = float(student_row_adj.get("Motivation", 0))
    return round(att, 1), round(exam, 1), round(hours, 1), round(mot, 1)


def get_weekly_trend_clean(student_row_adj, cohort_df, override_hours):
    if student_row_adj.empty:
        return pd.DataFrame()

    s = student_row_adj.iloc[0]

    # Student values
    student_hours = override_hours
    student_exam = float(s["ExamScore"])

    # Cohort values
    cohort_hours = cohort_df["StudyHours"].mean()
    cohort_exam = cohort_df["ExamScore"].mean()

    weeks = ["Week 2", "Week 4", "Week 8", "Week 14"]

    df = pd.DataFrame({
        "Week": weeks,
        "Student Exam Score": [student_exam * r for r in [0.75, 0.85, 0.95, 1.0]],
        "Cohort Exam Score":  [cohort_exam * r  for r in [0.75, 0.85, 0.95, 1.0]]
    })

    return df


def get_resource_distribution(student_row_adj):
    if student_row_adj is None or student_row_adj.empty:
        return pd.DataFrame()

    r = student_row_adj.iloc[0]
    avg_study = float(r.get("StudyHours", 0))
    online = float(r.get("EduTech", 0))
    tutoring = float(r.get("Extracurricular", 0))
    in_person = max(avg_study - online, 0)

    return pd.DataFrame({
        "Resource": ["Online Learning", "In-Person Classes", "Tutoring Sessions"],
        "Hours": [online, in_person, tutoring],
    })


def get_engagement_metrics(perf_df, student_id):
    row = perf_df[perf_df["Student_ID"] == student_id]
    if row.empty:
        row = perf_df.sample(1, random_state=0)
    r = row.iloc[0]

    ai_tutor = int(round(r.get("OnlineCourses", 0) / 2))
    study_group = int(round(r.get("Extracurricular", 0)))
    logins = int(round(r.get("Resources", 0) * 5))
    downloads = int(round(r.get("EduTech", 0) * 10))
    assign_comp = float(r.get("AssignmentCompletion", 0))
    stress_val = float(r.get("StressLevel", 0))
    forum_posts = int(round(r.get("Discussions", 0)))

    if stress_val >= 7:
        stress_label = "High"
    elif stress_val >= 4:
        stress_label = "Medium"
    else:
        stress_label = "Low"

    if assign_comp >= 70 and stress_val <= 6:
        status = "On Track"
    elif assign_comp >= 50:
        status = "Needs Attention"
    else:
        status = "At Risk"

    return {
        "AI_Tutor_Sessions": ai_tutor,
        "Study_Group_Sessions": study_group,
        "Dashboard_Logins": logins,
        "Resource_Downloads": downloads,
        "Assignment_Completion": round(assign_comp, 1),
        "Stress_Level_Label": stress_label,
        "Forum_Posts": forum_posts,
        "Overall_Status": status,
    }


def local_ai_agent(student_row_adj, risk_level):
    if student_row_adj is None or student_row_adj.empty:
        return ("No data available.",
                "We could not find your record.",
                "Please check with the program office.",
                "We could not compute an overall summary.")

    r = student_row_adj.iloc[0]
    att = float(r.get("Attendance", 0))
    study = float(r.get("StudyHours", 0))
    stress = float(r.get("StressLevel", 0))
    exams = float(r.get("ExamScore", 0))
    assign_comp = float(r.get("AssignmentCompletion", 0))

    if exams >= 75 and assign_comp >= 80:
        positive = "You are doing very well academically. Your exam scores and assignment completion look strong."
    elif exams >= 60:
        positive = "You are keeping up with the material. There is a good base to build on."
    else:
        positive = "You’ve made a start, and checking this dashboard is a good step forward."

    improv_parts = []
    if att < 70:
        improv_parts.append("Try to increase your class attendance.")
    if study < 10:
        improv_parts.append("Aim for at least 10–12 focused study hours per week.")
    if assign_comp < 70:
        improv_parts.append("Make sure you submit assignments on time.")
    improv = " ".join(improv_parts) if improv_parts else \
        "Keep refining your study routine and continue doing what already works."

    if stress >= 7:
        tip = "Your stress level looks high. Try short breaks, regular sleep, and reach out to your advisor if it feels overwhelming."
    elif stress >= 4:
        tip = "Your stress is moderate. Balance study time with rest and short walks between sessions."
    else:
        tip = "Your stress level looks manageable. Keep your current habits and protect your downtime."

    summary = (
        f"Based on your current data, your risk level is **{risk_level}**. "
        f"Attendance: **{att:.0f}%**, weekly study hours: **{study:.1f}**, "
        f"exam score: **{exams:.0f}**, and assignment completion: **{assign_comp:.0f}%**."
    )

    return positive, improv, tip, summary


def predict_risk(student_row_adj):
    if student_row_adj is None or student_row_adj.empty:
        return "Unknown"
    if risk_model is None:
        if "Risk_Level" in student_row_adj.columns:
            return str(student_row_adj["Risk_Level"].iloc[0])
        return "Medium"

    feature_cols = ['studyhours', 'attendance', 'resources', 'extracurricular', 'motivation', 'internet', 'gender', 'age', 'learningstyle', 'onlinecourses', 'discussions', 'assignmentcompletion', 'examscore', 'edutech', 'stresslevel', 'finalgrade', 'attendance_course', 'grade', 'credits', 'engagement_score', 'risk_level_enc']
    cols = [c for c in feature_cols if c in student_row_adj.columns]
    if not cols:
        return "Medium"

    X = student_row_adj[cols].astype(float).values.reshape(1, -1)
    pred = risk_model.predict(X)[0]
    return str(pred)


# -----------------------------------------------------------
# HOME PAGE
# -----------------------------------------------------------
def show_home():
    st.title("Silverleaf University – Student Success Dashboard")
    st.write(
        "Welcome! This tool helps you track your progress, understand your study habits, "
        "and get friendly AI-based suggestions to stay on track."
    )

    st.markdown("### Step 1: Find your name")
    name_to_id = dict(zip(profile_df["Name"], profile_df["Student_ID"]))
    selected_name = st.selectbox(
        "Search or select your name:",
        sorted(name_to_id.keys())
    )
    student_id = name_to_id[selected_name]

    # Pull defaults from profile
    stu_row = profile_df[profile_df["Student_ID"] == student_id].iloc[0]
    default_program = stu_row["Program"]
    default_year = stu_row["Year"]

    st.markdown("### Step 2: Confirm your program details")
    col1, col2, col3 = st.columns(3)
    with col1:
        semester = st.selectbox(
            "Semester",
            sorted(insights_df["Semester"].dropna().unique())
        )
    with col2:
        program = st.selectbox(
            "Program",
            sorted(profile_df["Program"].unique()),
            index=list(sorted(profile_df["Program"].unique())).index(default_program)
        )
    with col3:
        year = st.selectbox(
            "Year",
            sorted(profile_df["Year"].unique()),
            index=list(sorted(profile_df["Year"].unique())).index(default_year)
        )

    st.markdown("### Step 3: Enter your latest course grades")
    courses = get_courses_for_program(program)
    latest_grades = {}
    for c in courses:
        latest_grades[c] = st.number_input(
            f"{c} grade (0–100)",
            min_value=0,
            max_value=100,
            value=80,
            step=1
        )

    st.markdown("### Step 4: Quick study survey")
    survey_col1, survey_col2, survey_col3 = st.columns(3)
    with survey_col1:
        study_hours = st.slider("Study hours per week", 0, 60, 10)
    with survey_col2:
        attendance = st.slider("Attendance this semester (%)", 0, 100, 80)
    with survey_col3:
        motivation = st.slider("Motivation level (0–10)", 0, 10, 6)

    st.markdown("---")
    if st.button("Go to Dashboard", use_container_width=True):
        st.session_state["page"] = "dashboard"
        st.session_state["selected_student_id"] = student_id
        st.session_state["selected_student_name"] = selected_name
        st.session_state["selected_program"] = program
        st.session_state["selected_year"] = year
        st.session_state["selected_semester"] = semester
        st.session_state["latest_grades"] = latest_grades
        st.session_state["survey_study_hours"] = study_hours
        st.session_state["survey_attendance"] = attendance
        st.session_state["survey_motivation"] = motivation
        st.rerun()


# -----------------------------------------------------------
# DASHBOARD PAGE
# -----------------------------------------------------------
def show_dashboard():
    # safety: if someone lands here without state, send them home
    if st.session_state["selected_student_id"] is None:
        st.session_state["page"] = "home"
        st.rerun()

    selected_student_id = st.session_state["selected_student_id"]
    selected_name = st.session_state["selected_student_name"]
    selected_semester = st.session_state["selected_semester"]
    latest_grades = st.session_state["latest_grades"]

    # HEADER
    top_left, top_right = st.columns([4, 2])
    with top_left:
        st.subheader("Student Success Dashboard – Silverleaf University")
        st.caption("Track your progress. Discover insights. Improve outcomes.")

    with top_right:
        st.write(f"**Logged in as:** {selected_name}")
        if st.button("🔙 Back to Home", use_container_width=True):
            st.session_state["page"] = "home"
            st.rerun()

    # LAYOUT
    filters_col, main_col, right_col = st.columns([1.0, 3.0, 1.5])

    # ---------------------- FILTERS -------------------------
    with filters_col:
        st.markdown("### Filters")

        # -----------------------------
        # SEMESTER (synced with session)
        # -----------------------------
        semesters = sorted(insights_df["Semester"].dropna().unique())
        semester = st.selectbox(
            "Semester",
            options=semesters,
            index=semesters.index(st.session_state["selected_semester"])
        )

        # -----------------------------
        # PROGRAM (synced with session)
        # -----------------------------
        programs = ["All Programs"] + sorted(profile_df["Program"].unique())
        program = st.selectbox(
            "Program",
            options=programs,
            index=programs.index(st.session_state["selected_program"])
            if st.session_state["selected_program"] in programs else 0
        )

        # -----------------------------
        # YEAR (synced with session)
        # -----------------------------
        years = sorted(profile_df["Year"].unique())
        year = st.selectbox(
            "Year",
            options=years,
            index=years.index(st.session_state["selected_year"])
        )

        # -----------------------------
        # COURSE FILTER
        # -----------------------------
        courses_all = ["All Courses"] + sorted(enroll_df["Course_Name"].unique())
        course_filter = st.selectbox("Course", options=courses_all)

        # -----------------------------
        # RISK LEVEL FILTER (KEEP ORIGINAL)
        # -----------------------------
        risk_opts = ["All Levels", "Low", "Medium", "High"]
        risk_filter = st.selectbox("Risk Level", options=risk_opts)

        # -----------------------------
        # ATTENDANCE RANGE FILTER
        # -----------------------------
        att_min, att_max = st.slider(
            "Attendance Range (%)",
            0, 100,
            (0, 100)
        )

        # -----------------------------
        # STUDY HOURS (sync with survey)
        # -----------------------------
        st.session_state["survey_study_hours"] = st.slider(
            "Study Hours / Week",
            0, 60,
            value=st.session_state["survey_study_hours"]
        )

        # -----------------------------
        # ATTENDANCE override (sync)
        # -----------------------------
        st.session_state["survey_attendance"] = st.slider(
            "Override Attendance (%)",
            0, 100,
            value=st.session_state["survey_attendance"]
        )

        # -----------------------------
        # MOTIVATION override (sync)
        # -----------------------------
        st.session_state["survey_motivation"] = st.slider(
            "Motivation (0–10)",
            0, 10,
            value=st.session_state["survey_motivation"]
        )

    # =============================
    # BUILD COHORT USING FILTERS
    # =============================
    cohort = master_df.copy()

    # Filter semester
    cohort = cohort[cohort["Semester"] == semester]

    # Filter program
    if program != "All Programs":
        cohort = cohort[cohort["Program"] == program]

    # Filter course selection
    if course_filter != "All Courses":
        stu_course_ids = enroll_df[enroll_df["Course_Name"] == course_filter]["Student_ID"].unique()
        cohort = cohort[cohort["Student_ID"].isin(stu_course_ids)]

    # Filter attendance range
    if "Attendance" in cohort.columns:
        cohort = cohort[(cohort["Attendance"] >= att_min) & (cohort["Attendance"] <= att_max)]

    # Filter risk level
    if risk_filter != "All Levels":
        cohort = cohort[cohort["Risk_Level"] == risk_filter]

    # =======================================
    # BUILD STUDENT ROW (BASE + ADJUSTMENTS)
    # =======================================
    student_row = master_df[master_df["Student_ID"] == selected_student_id]
    if student_row.empty:
        # fallback to cohort
        student_row = cohort[cohort["Student_ID"] == selected_student_id]

    student_row_adj = student_row.copy()

    if not student_row_adj.empty:
        idx = student_row_adj.index[0]

        # override with SURVEY values (from home page OR dashboard)
        student_row_adj.loc[idx, "StudyHours"] = st.session_state["survey_study_hours"]
        student_row_adj.loc[idx, "Attendance"] = st.session_state["survey_attendance"]
        student_row_adj.loc[idx, "Motivation"] = st.session_state["survey_motivation"]

        # override exam score if latest course grades submitted
        if latest_grades:
            avg_new_exam = np.mean(list(latest_grades.values()))
            student_row_adj.loc[idx, "ExamScore"] = avg_new_exam


    # ---------------------- KPI ROW -------------------------
    with main_col:
        k1, k2, k3, k4 = st.columns(4)
        k_att, k_exam, k_hours, k_mot = compute_kpis(student_row_adj)

        with k1:
            st.markdown(
                f"""
                <div class="metric-card" style="background-color:#ff5f6d;">
                    <div class="metric-label">Attendance</div>
                    <div class="metric-value">{k_att}%</div>
                </div>
                """, unsafe_allow_html=True
            )
        with k2:
            st.markdown(
                f"""
                <div class="metric-card" style="background-color:#00BCD4;">
                    <div class="metric-label">Avg Exam Score</div>
                    <div class="metric-value">{k_exam}</div>
                </div>
                """, unsafe_allow_html=True
            )
        with k3:
            st.markdown(
                f"""
                <div class="metric-card" style="background-color:#ff9800;">
                    <div class="metric-label">Study Hours / Week</div>
                    <div class="metric-value">{k_hours}</div>
                </div>
                """, unsafe_allow_html=True
            )
        with k4:
            st.markdown(
                f"""
                <div class="metric-card" style="background-color:#3f51b5;">
                    <div class="metric-label">Motivation (0–10)</div>
                    <div class="metric-value">{k_mot}</div>
                </div>
                """, unsafe_allow_html=True
            )

    # ---------------------- MAIN VISUALS --------------------
    with main_col:
        # COURSE PERFORMANCE OVERVIEW
        st.markdown("#### Course Performance Overview")

        # historical grades for this student
        hist = enroll_df[enroll_df["Student_ID"] == selected_student_id][["Course_Name", "Grade"]]
        hist = hist.groupby("Course_Name", as_index=False)["Grade"].mean().rename(
            columns={"Grade": "Previous_Grade"}
        )

        # latest grades from survey
        latest_list = []
        for c, g in latest_grades.items():
            latest_list.append({"Course_Name": c, "Current_Grade": g})
        latest_df = pd.DataFrame(latest_list)

        course_perf = pd.merge(hist, latest_df, on="Course_Name", how="outer")

        if not course_perf.empty:
            # fill NaNs for plotting
            course_perf["Current_Grade"] = course_perf["Current_Grade"].fillna(0)

            fig_course = px.bar(
                course_perf,
                x="Course_Name",
                y=["Previous_Grade", "Current_Grade"],
                barmode="group"
            )
            fig_course.update_layout(
                showlegend=False,
                height=260,
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_course, use_container_width=True)
        else:
            st.info("No course data available for this student.")

        # WEEKLY STUDY HOURS & PERFORMANCE
        st.markdown("#### Weekly Study Hours & Performance Correlation")
        weekly_df = get_weekly_trend_clean(
            student_row_adj,
            cohort if not cohort.empty else master_df,
            override_hours=st.session_state["survey_study_hours"]
        )

        if not weekly_df.empty:
            fig_week = px.line(
                weekly_df,
                x="Week",
                y=["Student Exam Score", "Cohort Exam Score"],
                markers=True
            )
            fig_week.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=30, b=10),
                showlegend=True
            )

            st.plotly_chart(fig_week, use_container_width=True)

        else:
            st.info("Not enough data to build a weekly trend.")

        # STUDY RESOURCES + ENGAGEMENT
        col_res, col_eng = st.columns(2)

        with col_res:
            st.markdown("#### Study Resources Distribution")
            res_df = get_resource_distribution(student_row_adj)
            if not res_df.empty:
                fig_res = px.pie(
                    res_df,
                    names="Resource",
                    values="Hours",
                    hole=0.5,
                )
                fig_res.update_layout(
                    height=260,
                    margin=dict(l=10, r=10, t=30, b=10)
                )
                st.plotly_chart(fig_res, use_container_width=True)
            else:
                st.info("No resource data available.")

        with col_eng:
            st.markdown("#### Academic Engagement Metrics")

            eng = get_engagement_metrics(perf_df, selected_student_id)

            for label, val in [
                ("AI Tutor Sessions", eng["AI_Tutor_Sessions"]),
                ("Study Group Sessions", eng["Study_Group_Sessions"]),
                ("Dashboard Logins", eng["Dashboard_Logins"]),
                ("Resource Downloads", eng["Resource_Downloads"]),
                ("Assignment Completion", f"{eng['Assignment_Completion']}%"),
                ("Stress Level", eng["Stress_Level_Label"]),
                ("Forum Participation", f"{eng['Forum_Posts']} posts"),
                ("Overall Progress", eng["Overall_Status"]),
            ]:
                st.markdown(
                    f"""
                    <div style="
                        display:flex;
                        justify-content:space-between;
                        padding:6px 0;
                        border-bottom:1px solid #eee;
                        font-size:15px;
                        color:#1a1a1a;">
                        <span style="font-weight:600;">{label}</span>
                        <span style="font-weight:600; color:#1b4ed8;">{val}</span>
                    </div>
                    """,
                    unsafe_allow_html=True
                )        

    # ---------------------- RIGHT COLUMN: ANALYZE -----------
    with right_col:
        st.markdown(
            """
            <div class="analyze-card">
                <div style="font-weight:600; font-size:13px; margin-bottom:8px;">
                    Analyze My Performance
                </div>
                <div style="font-size:11px; margin-bottom:10px;">
                    Click the button below to generate personalized academic insights
                    based on your latest grades and survey responses.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        clicked = st.button("🔍 Analyze My Performance", use_container_width=True)

        st.markdown("<br/>", unsafe_allow_html=True)
        st.markdown('<div class="panel-header">🎓 AI-Generated Academic Insights</div>',
                    unsafe_allow_html=True)

        if not clicked:
            st.info("No analysis yet. Click **Analyze My Performance** to view your insights.")
        else:
            risk_level = predict_risk(student_row_adj)
            pos_msg, improv_msg, tip_msg, summary_msg = local_ai_agent(
                student_row_adj, risk_level
            )

            st.markdown(
                f"""
                <div class="insight-card">
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
                <div class="insight-card">
                    <div class="insight-title">📊 Overall Summary</div>
                    <div class="insight-body">{summary_msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="footer-container">
            <div class="footer-note">
                This dashboard is a tool for student academic success insights.
            </div>
            <div class="footer-copy">
                © 2025 Silverleaf University · All Rights Reserved · Student Success Dashboard Prototype
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -----------------------------------------------------------
# ROUTER
# -----------------------------------------------------------
if st.session_state["page"] == "home":
    show_home()
else:
    show_dashboard()