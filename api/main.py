from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
import io
from PyPDF2 import PdfReader
import re
from rapidfuzz import fuzz

app = FastAPI()

# --- minimal skill/data definitions (keep synced with Streamlit app if needed) ---
skills_list = [
    "python", "java", "c++", "sql", "html", "css", "javascript",
    "machine learning", "deep learning", "data analysis",
    "pandas", "numpy", "tensorflow", "flask", "react",
    "mongodb", "excel", "power bi"
]

skill_aliases = {
    "ml": "machine learning",
    "machine-learning": "machine learning",
    "deep-learning": "deep learning",
    "js": "javascript",
    "py": "python",
    "c": "c++",
    "powerbi": "power bi",
    "nlp": "natural language processing",
}

skill_ontology = {
    "machine learning": ["ml", "supervised learning", "unsupervised learning", "deep learning"],
    "data analysis": ["pandas", "numpy", "data visualization", "excel"],
    "web development": ["html", "css", "javascript", "react", "flask"],
    "databases": ["sql", "mongodb"],
}


def extract_text_from_pdf_bytes(data: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(data))
        text = ""
        for p in reader.pages:
            t = p.extract_text() or ""
            text += t + "\n"
        return text
    except Exception:
        return ""


def extract_skills(text: str):
    found = set()
    text = (text or "").lower()
    sentences = [s.strip() for s in re.split(r'[\n\r\.]+' , text) if s.strip()]
    for skill in skills_list:
        canonical = skill.lower()
        if re.search(r'\b' + re.escape(canonical) + r'\b', text):
            found.add(skill)
            continue
        for alias, target in skill_aliases.items():
            if target == canonical:
                if re.search(r'\b' + re.escape(alias) + r'\b', text):
                    found.add(skill)
                    break
        if skill in found:
            continue
        for s in sentences:
            if fuzz.partial_ratio(canonical, s) >= 85:
                found.add(skill)
                break

    # map ontology parents
    for parent, terms in skill_ontology.items():
        parent_lower = parent.lower()
        if any(parent_lower == f.lower() for f in found):
            continue
        for t in terms:
            t_lower = t.lower()
            if re.search(r'\b' + re.escape(t_lower) + r'\b', text):
                found.add(parent)
                break
    return sorted(found)


def parse_job_description(jd_text: str):
    lines = [l.strip() for l in (jd_text or "").splitlines() if l.strip()]
    required = set()
    optional = set()
    current = None
    for line in lines:
        low = line.lower()
        if low.startswith("required") or "must have" in low:
            current = "required"
            continue
        if low.startswith("optional") or "nice to have" in low:
            current = "optional"
            continue
        found = extract_skills(line)
        if current == "required":
            required.update(found)
        elif current == "optional":
            optional.update(found)
        else:
            # heuristics: lines with 'required' keywords
            if "required" in low or "must" in low:
                required.update(found)
            else:
                required.update(found)
    return {"required": sorted(required), "optional": sorted(optional)}


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/match")
async def match_resume(file: UploadFile = File(...), job_description: Optional[str] = Form("")):
    payload = await file.read()
    resume_text = extract_text_from_pdf_bytes(payload)
    if not resume_text:
        return JSONResponse({"error": "Could not extract text from PDF"}, status_code=400)

    skills = extract_skills(resume_text)
    jd_parsed = parse_job_description(job_description)

    result = {
        "resume_name": file.filename,
        "resume_preview": resume_text[:1000],
        "extracted_skills": skills,
        "jd_required": jd_parsed.get("required", []),
        "jd_optional": jd_parsed.get("optional", []),
        "notes": "This lightweight endpoint does not compute embeddings. For semantic matching, call an external embedding service or host a separate model server."
    }

    return result
