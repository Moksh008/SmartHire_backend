import io
import re
import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
import numpy as np

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🧠",
    layout="wide"
)

# ─────────────────────────────────────────────
# LOAD BERT MODEL (cached so it loads once)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(io.BytesIO(uploaded_file.read()))
    text = ""
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9 .,]', '', text)
    return text.strip()


def chunk_text(text, chunk_size=80):
    words = text.split()
    chunks = []
    step = chunk_size // 2
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
    return chunks


def get_embeddings(chunks):
    return model.encode(chunks, batch_size=32, show_progress_bar=False)


def bert_similarity_score(resume_text, jd_text):
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    resume_chunks = chunk_text(resume_text)
    jd_chunks = chunk_text(jd_text)
    if not resume_chunks or not jd_chunks:
        return 0.0
    resume_emb = get_embeddings(resume_chunks)
    jd_emb = get_embeddings(jd_chunks)
    sim_matrix = cosine_similarity(resume_emb, jd_emb)
    best_per_jd = sim_matrix.max(axis=0)
    return float(best_per_jd.mean())


def calculate_fuzzy_score(resume_text, jd_text):
    return fuzz.partial_ratio(resume_text[:5000], jd_text[:5000])


# ─────────────────────────────────────────────
# SKILL EXTRACTION
# ─────────────────────────────────────────────

# Required skills signals in JD text
REQUIRED_SIGNALS = [
    "required", "must have", "must-have", "essential", "mandatory",
    "you must", "we require", "minimum requirement", "necessary"
]
OPTIONAL_SIGNALS = [
    "preferred", "nice to have", "nice-to-have", "bonus", "plus",
    "advantage", "desirable", "good to have", "optional", "beneficial"
]

SKILL_KEYWORDS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "r", "go", "rust", "scala",
    "machine learning", "deep learning", "nlp", "bert", "gpt", "llm", "ai", "artificial intelligence",
    "computer vision", "reinforcement learning", "neural network", "transformer",
    "tensorflow", "pytorch", "keras", "scikit-learn", "xgboost", "lightgbm", "catboost",
    "pandas", "numpy", "matplotlib", "seaborn", "plotly", "scipy", "statsmodels",
    "data engineering", "etl", "data pipelines", "airflow", "apache spark", "spark", "kafka",
    "aws", "gcp", "azure", "cloud", "docker", "kubernetes", "ci/cd", "devops", "terraform",
    "react", "node.js", "fastapi", "flask", "django", "spring boot", "express",
    "postgresql", "mysql", "mongodb", "redis", "elasticsearch", "cassandra", "snowflake",
    "git", "linux", "rest api", "graphql", "microservices", "grpc",
    "streamlit", "tableau", "power bi", "looker", "qlik",
    "data analysis", "data science", "data visualization", "business intelligence",
    "communication", "leadership", "teamwork", "problem solving", "agile", "scrum",
    "project management", "product management", "stakeholder management",
    "excel", "vba", "sas", "matlab", "hadoop", "hive", "pig",
    "penetration testing", "cybersecurity", "network security", "ethical hacking",
    "nlp", "text mining", "feature engineering", "a/b testing", "experimentation",
    "html", "css", "sass", "webpack", "git", "bash", "shell", "powershell"
]

DEGREE_KEYWORDS = {
    "phd": 5, "doctorate": 5, "ph.d": 5,
    "master": 4, "m.s.": 4, "msc": 4, "m.tech": 4, "mba": 4, "m.e.": 4,
    "bachelor": 3, "b.s.": 3, "bsc": 3, "b.tech": 3, "b.e.": 3, "undergraduate": 3,
    "associate": 2, "diploma": 2,
    "high school": 1, "secondary": 1
}

CERT_KEYWORDS = [
    "aws certified", "google cloud", "azure certified", "cka", "ckad",
    "pmp", "scrum master", "csm", "cissp", "ceh", "comptia",
    "tensorflow certificate", "databricks", "snowflake", "tableau certified",
    "coursera", "udacity", "edx", "certification", "certified"
]

JOB_TITLE_KEYWORDS = [
    "software engineer", "data scientist", "data engineer", "ml engineer", "machine learning engineer",
    "backend developer", "frontend developer", "full stack", "devops engineer", "cloud engineer",
    "product manager", "project manager", "business analyst", "data analyst",
    "security engineer", "cybersecurity", "ai engineer", "nlp engineer",
    "research scientist", "research engineer", "applied scientist",
    "architect", "tech lead", "senior engineer", "junior engineer",
    "intern", "associate", "lead", "principal", "staff engineer", "manager"
]


def extract_skills_from_text(text):
    text_lower = text.lower()
    found = [skill for skill in SKILL_KEYWORDS if skill in text_lower]
    return list(set(found))


def classify_jd_skills(jd_text):
    """
    Split JD skills into required vs optional by parsing context sentences.
    Returns: (required_skills, optional_skills)
    """
    required_skills = []
    optional_skills = []
    all_jd_skills = extract_skills_from_text(jd_text)

    jd_lower = jd_text.lower()
    sentences = re.split(r'[.\n;]', jd_lower)

    skill_classification = {}
    for skill in all_jd_skills:
        skill_classification[skill] = "required"  # default

    for sentence in sentences:
        is_required = any(sig in sentence for sig in REQUIRED_SIGNALS)
        is_optional = any(sig in sentence for sig in OPTIONAL_SIGNALS)
        for skill in all_jd_skills:
            if skill in sentence:
                if is_optional:
                    skill_classification[skill] = "optional"
                elif is_required:
                    skill_classification[skill] = "required"

    for skill, label in skill_classification.items():
        if label == "required":
            required_skills.append(skill)
        else:
            optional_skills.append(skill)

    return required_skills, optional_skills


def match_skills_bert(resume_skills, jd_skills, threshold=0.65):
    if not resume_skills or not jd_skills:
        return [], jd_skills
    jd_emb = get_embeddings(jd_skills)
    res_emb = get_embeddings(resume_skills)
    sim_matrix = cosine_similarity(res_emb, jd_emb)
    matched = []
    missing = []
    for j, jd_skill in enumerate(jd_skills):
        best_score = sim_matrix[:, j].max()
        if best_score >= threshold:
            matched.append((jd_skill, float(best_score)))
        else:
            missing.append(jd_skill)
    return matched, missing


# ─────────────────────────────────────────────
# EXPERIENCE PARSER
# ─────────────────────────────────────────────

def extract_years_of_experience(text):
    """
    Extract total years of experience from resume text.
    Looks for patterns like '5 years', '3+ years', date ranges, etc.
    """
    text_lower = text.lower()
    years = []

    # Pattern: "X years of experience" or "X+ years"
    pattern1 = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)', text_lower)
    years.extend([int(y) for y in pattern1])

    # Pattern: date ranges like "2019 - 2023" or "Jan 2018 – Dec 2022"
    pattern2 = re.findall(r'(20\d{2}|19\d{2})\s*[-–—to]+\s*(20\d{2}|19\d{2}|present|current)', text_lower)
    current_year = 2024
    for start, end in pattern2:
        try:
            s = int(start)
            e = current_year if end in ('present', 'current') else int(end)
            diff = e - s
            if 0 < diff <= 50:
                years.append(diff)
        except:
            pass

    if not years:
        return 0

    # Avoid double-counting overlapping ranges
    return min(sum(years), 40)  # cap at 40 to avoid overflow


def extract_required_experience(jd_text):
    """Extract required years from JD."""
    patterns = re.findall(r'(\d+)\+?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)', jd_text.lower())
    if patterns:
        return max([int(p) for p in patterns])
    return 0


def extract_job_titles(text):
    text_lower = text.lower()
    found = [title for title in JOB_TITLE_KEYWORDS if title in text_lower]
    return found


def score_experience(resume_text, jd_text):
    """
    Score experience out of 100:
    - Years match: 50 pts
    - Role title relevance: 30 pts (BERT)
    - Seniority alignment: 20 pts
    """
    resume_years = extract_years_of_experience(resume_text)
    required_years = extract_required_experience(jd_text)

    # Years score (50 pts)
    if required_years == 0:
        years_score = 50  # no requirement stated, give full
    elif resume_years >= required_years:
        years_score = 50
    else:
        years_score = round((resume_years / required_years) * 50)

    # Role relevance via BERT (30 pts)
    resume_exp_section = extract_section(resume_text.lower(), SECTION_HEADERS["experience"])
    role_score_raw = bert_similarity_score(resume_exp_section or resume_text, jd_text) if resume_exp_section else 0
    role_score = round(role_score_raw * 30)

    # Seniority alignment (20 pts)
    jd_titles = extract_job_titles(jd_text)
    resume_titles = extract_job_titles(resume_text)
    seniority_keywords = {
        "senior": 3, "lead": 3, "principal": 4, "staff": 4, "architect": 4,
        "junior": 1, "intern": 0, "associate": 2, "manager": 3, "director": 4
    }
    jd_level = 2  # default mid-level
    resume_level = 2

    for kw, level in seniority_keywords.items():
        if kw in jd_text.lower():
            jd_level = level
        if kw in resume_text.lower():
            resume_level = level

    if resume_level >= jd_level:
        seniority_score = 20
    else:
        gap = jd_level - resume_level
        seniority_score = max(0, 20 - gap * 5)

    total = min(years_score + role_score + seniority_score, 100)
    return total, resume_years, required_years, resume_level, jd_level


# ─────────────────────────────────────────────
# EDUCATION PARSER
# ─────────────────────────────────────────────

def score_education(resume_text, jd_text):
    """
    Score education out of 100:
    - Degree level match: 60 pts
    - Field relevance (BERT): 40 pts
    """
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()

    resume_degree_level = 0
    resume_degree_name = "Not detected"
    for degree, level in sorted(DEGREE_KEYWORDS.items(), key=lambda x: -x[1]):
        if degree in resume_lower:
            resume_degree_level = level
            resume_degree_name = degree.title()
            break

    jd_degree_level = 0
    for degree, level in sorted(DEGREE_KEYWORDS.items(), key=lambda x: -x[1]):
        if degree in jd_lower:
            jd_degree_level = level
            break

    if jd_degree_level == 0:
        jd_degree_level = 3  # default: assume bachelor's required

    if resume_degree_level >= jd_degree_level:
        degree_score = 60
    else:
        gap = jd_degree_level - resume_degree_level
        degree_score = max(0, 60 - gap * 15)

    # Field relevance via BERT
    edu_section = extract_section(resume_text.lower(), SECTION_HEADERS["education"])
    if edu_section:
        field_score_raw = bert_similarity_score(edu_section, jd_text)
        field_score = round(field_score_raw * 40)
    else:
        field_score = 20  # partial credit if section not found

    total = min(degree_score + field_score, 100)
    return total, resume_degree_name, resume_degree_level, jd_degree_level


# ─────────────────────────────────────────────
# CERTIFICATION SCORER
# ─────────────────────────────────────────────

def score_certifications(resume_text, jd_text):
    """Score certifications: presence in resume vs. JD mentions (0-100)."""
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()

    jd_certs = [c for c in CERT_KEYWORDS if c in jd_lower]
    resume_certs = [c for c in CERT_KEYWORDS if c in resume_lower]

    if not jd_certs:
        # No certs required; bonus for having any
        if resume_certs:
            return 80, resume_certs, []
        return 70, [], []

    matched = [c for c in jd_certs if c in resume_lower]
    missing = [c for c in jd_certs if c not in resume_lower]
    score = round((len(matched) / len(jd_certs)) * 100)
    return score, matched, missing


# ─────────────────────────────────────────────
# WEIGHTED ATS SCORE COMPUTATION
# ─────────────────────────────────────────────

ATS_WEIGHTS = {
    "skills_required":   0.30,   # Required skills match
    "skills_optional":   0.10,   # Optional/preferred skills
    "semantic":          0.20,   # Overall semantic similarity
    "experience":        0.25,   # Experience depth + years
    "education":         0.10,   # Degree level + relevance
    "certifications":    0.05,   # Certifications
}

def compute_ats_score(
    required_skill_score,
    optional_skill_score,
    semantic_score,
    experience_score,
    education_score,
    cert_score
):
    """
    Weighted ATS score based on realistic recruiter priorities.
    Returns 0–100.
    """
    raw = (
        ATS_WEIGHTS["skills_required"] * required_skill_score +
        ATS_WEIGHTS["skills_optional"] * optional_skill_score +
        ATS_WEIGHTS["semantic"] * semantic_score +
        ATS_WEIGHTS["experience"] * experience_score +
        ATS_WEIGHTS["education"] * education_score +
        ATS_WEIGHTS["certifications"] * cert_score
    )
    return round(min(raw, 100), 2)


# ─────────────────────────────────────────────
# SECTION EXTRACTION
# ─────────────────────────────────────────────
SECTION_HEADERS = {
    "skills":     ["skills", "technical skills", "key skills", "tools", "technologies"],
    "experience": ["experience", "work experience", "employment", "professional experience"],
    "education":  ["education", "qualifications", "academic"],
    "projects":   ["projects", "personal projects", "academic projects"],
    "summary":    ["summary", "objective", "profile", "about"]
}

def extract_section(text, section_keywords):
    lines = text.lower().split('\n')
    in_section = False
    section_text = []
    for line in lines:
        if any(kw in line for kw in section_keywords):
            in_section = True
            continue
        if in_section:
            if any(
                any(kw in line for kw in v)
                for k, v in SECTION_HEADERS.items()
                if not any(kw in line for kw in section_keywords)
            ):
                break
            section_text.append(line)
    return " ".join(section_text).strip()


# ─────────────────────────────────────────────
# VISUALIZATION (UNCHANGED)
# ─────────────────────────────────────────────
def draw_score_gauge(score):
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(polar=False))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    theta = np.linspace(np.pi, 0, 200)
    ax.plot(np.cos(theta), np.sin(theta), color='#2a2a3a', linewidth=20)
    score_angle = np.pi - (score / 100) * np.pi
    theta_score = np.linspace(np.pi, score_angle, 200)
    color = '#00e5ff' if score >= 70 else '#ffb300' if score >= 40 else '#f44336'
    ax.plot(np.cos(theta_score), np.sin(theta_score), color=color, linewidth=20)
    ax.text(0, -0.15, f"{score:.1f}%", ha='center', va='center',
            fontsize=30, color='white', fontweight='bold')
    ax.text(0, -0.4, get_label(score), ha='center', va='center',
            fontsize=12, color=color)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.6, 1.2)
    ax.axis('off')
    return fig


def get_label(score):
    if score >= 80: return "Excellent Match"
    elif score >= 65: return "Strong Match"
    elif score >= 45: return "Moderate Match"
    else: return "Weak Match"


def draw_skill_bar_chart(matched):
    if not matched:
        return None
    labels = [m[0].title() for m in matched]
    scores = [m[1] * 100 for m in matched]
    colors = ['#00e5ff' if s >= 80 else '#4fc3f7' if s >= 65 else '#ffb300' for s in scores]
    fig, ax = plt.subplots(figsize=(7, max(3, len(labels) * 0.5)))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    bars = ax.barh(labels, scores, color=colors, height=0.6)
    ax.set_xlim(0, 110)
    ax.set_xlabel("Match %", color='white')
    ax.tick_params(colors='white')
    ax.spines[:].set_visible(False)
    ax.set_title("Matched Skills", color='white', pad=10)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{score:.0f}%", va='center', color='white', fontsize=9)
    plt.tight_layout()
    return fig


def draw_component_radar(scores_dict):
    """Radar chart showing all ATS component scores."""
    labels = list(scores_dict.keys())
    values = list(scores_dict.values())
    N = len(labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values_plot = values + [values[0]]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#0f0f1a')
    ax.plot(angles, values_plot, 'o-', linewidth=2, color='#00e5ff')
    ax.fill(angles, values_plot, alpha=0.2, color='#00e5ff')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, color='white', fontsize=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], color='#555', fontsize=7)
    ax.grid(color='#2a2a4a', linestyle='--', linewidth=0.8)
    ax.spines['polar'].set_color('#2a2a4a')
    ax.set_title("ATS Component Scores", color='white', pad=20, fontsize=12)
    return fig


# ─────────────────────────────────────────────
# STREAMLIT UI (UNCHANGED STYLING)
# ─────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono&family=Sora:wght@400;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Sora', sans-serif;
        background-color: #0f0f1a;
        color: #e0e0e0;
    }
    .stTextArea textarea {
        background-color: #1a1a2e !important;
        color: #e0e0e0 !important;
        border: 1px solid #2a2a4a !important;
        border-radius: 8px !important;
        font-family: 'Space Mono', monospace !important;
        font-size: 13px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #00e5ff, #7b2ff7);
        color: white;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 16px;
    }
    .metric-box {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .skill-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 13px;
        margin: 4px;
        font-family: 'Space Mono', monospace;
    }
    .matched { background: #003d2e; color: #00e5b0; border: 1px solid #00e5b0; }
    .missing { background: #3d0000; color: #ff5252; border: 1px solid #ff5252; }
    .optional-tag { background: #1a1a00; color: #ffb300; border: 1px solid #ffb300; }
    .weight-badge {
        display: inline-block;
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 11px;
        color: #888;
        margin-left: 6px;
        font-family: 'Space Mono', monospace;
    }
    </style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
    <h1 style='font-family:Sora;font-size:2.4rem;
    background:linear-gradient(90deg,#00e5ff,#7b2ff7);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;'>
    AI Resume Screener
    </h1>
    <p style='color:#888;margin-top:0;font-size:15px;'>
    BERT-powered contextual matching — not just keywords
    </p>
    <hr style='border-color:#2a2a4a;margin:16px 0;'>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# INPUT SECTION
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Upload Resume (PDF)")
    resume_file = st.file_uploader("", type=["pdf"], key="resume")

with col2:
    st.markdown("#### Job Description")
    jd_text_input = st.text_area("Paste the Job Description here", height=220, key="jd")

threshold = st.slider(
    "Skill Match Threshold",
    min_value=0.40, max_value=0.90, value=0.65, step=0.05,
    help="Higher = stricter skill matching. 0.65 is recommended."
)

analyze_btn = st.button("Analyze Resume")

# ─────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────
if analyze_btn:
    if not resume_file or not jd_text_input.strip():
        st.warning("Please upload a resume PDF and paste a job description.")
    else:
        with st.spinner("Running BERT analysis..."):

            # 1. Extract & clean
            raw_resume = extract_text_from_pdf(resume_file)
            resume_clean = clean_text(raw_resume)
            jd_clean = clean_text(jd_text_input)

            if len(resume_clean) < 50:
                st.error("Could not extract enough text from the PDF. Try a text-based PDF.")
                st.stop()

            # 2. Overall BERT semantic score (0–100)
            bert_score_raw = bert_similarity_score(resume_clean, jd_clean)
            semantic_score = round(bert_score_raw * 100, 2)

            # 3. Classify JD skills: required vs optional
            required_jd_skills, optional_jd_skills = classify_jd_skills(jd_text_input)
            resume_skills = extract_skills_from_text(resume_clean)

            # 4. Match required skills (weighted heavier)
            matched_required, missing_required = match_skills_bert(
                resume_skills, required_jd_skills, threshold=threshold
            )
            if required_jd_skills:
                required_skill_score = round((len(matched_required) / len(required_jd_skills)) * 100)
            else:
                required_skill_score = semantic_score  # fallback

            # 5. Match optional skills
            matched_optional, missing_optional = match_skills_bert(
                resume_skills, optional_jd_skills, threshold=threshold
            )
            if optional_jd_skills:
                optional_skill_score = round((len(matched_optional) / len(optional_jd_skills)) * 100)
            else:
                optional_skill_score = 50  # neutral if none specified

            # All matched/missing for display
            all_matched = matched_required + matched_optional
            all_missing = missing_required + missing_optional

            # 6. Experience score
            exp_score, resume_years, required_years, resume_level, jd_level = score_experience(
                raw_resume, jd_text_input
            )

            # 7. Education score
            edu_score, degree_name, resume_deg_level, jd_deg_level = score_education(
                raw_resume, jd_text_input
            )

            # 8. Certification score
            cert_score, matched_certs, missing_certs = score_certifications(
                raw_resume, jd_text_input
            )

            # 9. Weighted final ATS score
            final_ats_score = compute_ats_score(
                required_skill_score=required_skill_score,
                optional_skill_score=optional_skill_score,
                semantic_score=semantic_score,
                experience_score=exp_score,
                education_score=edu_score,
                cert_score=cert_score
            )

        # ─────────────────────────────────────────────
        # RESULTS
        # ─────────────────────────────────────────────
        st.markdown("<hr style='border-color:#2a2a4a;'>", unsafe_allow_html=True)
        st.markdown("## Analysis Results")

        g_col, m_col = st.columns([1.2, 1.8])

        with g_col:
            fig_gauge = draw_score_gauge(final_ats_score)
            st.pyplot(fig_gauge, use_container_width=True)
            plt.close()

        with m_col:
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
            st.markdown(f"""
                <div class='metric-box'>
                    <p style='margin:0;color:#888;font-size:12px;'>Required Skills Match
                        <span class='weight-badge'>30% weight</span></p>
                    <p style='margin:0;font-size:22px;font-weight:bold;color:#00e5ff;'>{required_skill_score}%</p>
                </div>
                <div class='metric-box'>
                    <p style='margin:0;color:#888;font-size:12px;'>Experience Score
                        <span class='weight-badge'>25% weight</span></p>
                    <p style='margin:0;font-size:22px;font-weight:bold;color:#7b2ff7;'>{exp_score}%
                        <span style='font-size:13px;color:#888;'> · {resume_years}y detected / {required_years}y required</span></p>
                </div>
                <div class='metric-box'>
                    <p style='margin:0;color:#888;font-size:12px;'>Semantic Similarity
                        <span class='weight-badge'>20% weight</span></p>
                    <p style='margin:0;font-size:22px;font-weight:bold;color:#ffb300;'>{semantic_score:.1f}%</p>
                </div>
                <div class='metric-box'>
                    <p style='margin:0;color:#888;font-size:12px;'>Education Score
                        <span class='weight-badge'>10% weight</span></p>
                    <p style='margin:0;font-size:22px;font-weight:bold;color:#4fc3f7;'>{edu_score}%
                        <span style='font-size:13px;color:#888;'> · {degree_name}</span></p>
                </div>
            """, unsafe_allow_html=True)

        # Radar chart of all components
        st.markdown("---")
        radar_col, cert_col = st.columns([1.2, 1])

        with radar_col:
            st.markdown("#### Score Components Radar")
            radar_data = {
                "Req. Skills": required_skill_score,
                "Opt. Skills": optional_skill_score,
                "Semantic": semantic_score,
                "Experience": exp_score,
                "Education": edu_score,
                "Certs": cert_score
            }
            fig_radar = draw_component_radar(radar_data)
            st.pyplot(fig_radar, use_container_width=True)
            plt.close()

        with cert_col:
            st.markdown("#### Certifications")
            if matched_certs:
                st.markdown("**Matched:**")
                for c in matched_certs:
                    st.markdown(f"<span class='skill-tag matched'>{c.title()}</span>", unsafe_allow_html=True)
            if missing_certs:
                st.markdown("**Missing from JD:**")
                for c in missing_certs:
                    st.markdown(f"<span class='skill-tag missing'>{c.title()}</span>", unsafe_allow_html=True)
            if not matched_certs and not missing_certs:
                st.info("No specific certifications detected in JD.")
            st.markdown(f"""
                <div class='metric-box' style='margin-top:12px;'>
                    <p style='margin:0;color:#888;font-size:12px;'>Cert Score
                        <span class='weight-badge'>5% weight</span></p>
                    <p style='margin:0;font-size:22px;font-weight:bold;color:#00e5b0;'>{cert_score}%</p>
                </div>
            """, unsafe_allow_html=True)

        # Skills breakdown
        st.markdown("---")
        s1, s2 = st.columns(2)

        with s1:
            st.markdown(f"#### Required Skills ({len(matched_required)} / {len(required_jd_skills)} matched)")
            if matched_required:
                tags = "".join([
                    f"<span class='skill-tag matched'>{s[0].title()} · {s[1]*100:.0f}%</span>"
                    for s in matched_required
                ])
                st.markdown(tags, unsafe_allow_html=True)
            if missing_required:
                st.markdown("**Missing required:**")
                tags = "".join([
                    f"<span class='skill-tag missing'>{s.title()}</span>"
                    for s in missing_required
                ])
                st.markdown(tags, unsafe_allow_html=True)

        with s2:
            st.markdown(f"#### Optional / Preferred Skills ({len(matched_optional)} / {len(optional_jd_skills)} matched)")
            if matched_optional:
                tags = "".join([
                    f"<span class='skill-tag optional-tag'>{s[0].title()} · {s[1]*100:.0f}%</span>"
                    for s in matched_optional
                ])
                st.markdown(tags, unsafe_allow_html=True)
            if missing_optional:
                st.markdown("**Not found (optional):**")
                tags = "".join([
                    f"<span class='skill-tag missing'>{s.title()}</span>"
                    for s in missing_optional
                ])
                st.markdown(tags, unsafe_allow_html=True)
            if not optional_jd_skills:
                st.info("No optional skills detected in JD.")

        # Skill bar chart
        if all_matched:
            st.markdown("---")
            st.markdown("#### Skill Match Breakdown")
            fig_bar = draw_skill_bar_chart(all_matched)
            if fig_bar:
                st.pyplot(fig_bar, use_container_width=True)
                plt.close()

        # Insight
        st.markdown("---")
        st.markdown("#### Insights")

        insights = []

        if required_skill_score < 50:
            missing_preview = ', '.join(missing_required[:3]) if missing_required else 'N/A'
            insights.append(f"⚠️ <b>Critical gap:</b> Only {required_skill_score}% of required skills matched. Missing: {missing_preview}.")
        if resume_years < required_years and required_years > 0:
            insights.append(f"⚠️ <b>Experience gap:</b> Resume shows ~{resume_years} years, JD requires {required_years}+ years.")
        if edu_score < 50:
            insights.append(f"⚠️ <b>Education gap:</b> Detected degree ({degree_name}) may be below JD requirements.")
        if optional_skill_score > 70:
            insights.append(f"✅ <b>Strong preferred skills match ({optional_skill_score}%).</b> Candidate has desirable bonus skills.")
        if final_ats_score >= 80:
            insights.append("🔥 <b>Strong overall candidate.</b> Resume aligns well across all key parameters.")
        elif final_ats_score >= 60:
            insights.append("👍 <b>Good match.</b> Solid alignment on most parameters with some gaps.")
        elif final_ats_score >= 40:
            insights.append("⚡ <b>Moderate match.</b> Relevant background but notable gaps in key areas.")
        else:
            insights.append("❌ <b>Weak match.</b> Resume does not align well with this role's requirements.")

        for insight in insights:
            st.markdown(f"<div class='metric-box'>{insight}</div>", unsafe_allow_html=True)

        # Summary table
        st.markdown("---")
        st.markdown("#### Score Summary")
        summary_df = pd.DataFrame({
            "Parameter": [
                "🏆 Final ATS Score",
                "✅ Required Skills Match",
                "⭐ Optional Skills Match",
                "🧠 Semantic Similarity",
                "💼 Experience Score",
                "🎓 Education Score",
                "📜 Certification Score"
            ],
            "Score": [
                f"{final_ats_score:.1f}%",
                f"{required_skill_score}%",
                f"{optional_skill_score}%",
                f"{semantic_score:.1f}%",
                f"{exp_score}%",
                f"{edu_score}%",
                f"{cert_score}%"
            ],
            "Weight": [
                "Composite",
                "30%",
                "10%",
                "20%",
                "25%",
                "10%",
                "5%"
            ]
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)