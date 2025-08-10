import os, re, json
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- Safe JSON extraction (no eval) ----------
JSON_ARRAY_RE = re.compile(r"\[\s*(?:\{.*?\}\s*,\s*)*\{.*?\}\s*\]", re.S)
JSON_OBJECT_RE = re.compile(r"\{\s*\"[^\"]+\"\s*:\s*.*\}", re.S)

def parse_json_strict(text, want="array"):
    text = (text or "").strip()
    m = JSON_ARRAY_RE.search(text) if want == "array" else JSON_OBJECT_RE.search(text)
    if m:
        return json.loads(m.group(0))
    return json.loads(text)

# ---------- YouTube search: whole-unit/whole-topic review ----------
def search_youtube_precise(subject, topic_title):
    """
    Prefer: AP -> 'AP <subject> Unit N review' matches.
            Normal -> '<topic> unit/chapter/full review' matches.
    Returns first strong match, else first result, else "".
    """
    from youtubesearchpython import VideosSearch

    ap = bool(re.search(r"\bap\b", (subject or "").lower()))
    unit_num = None
    m = re.search(r"\bunit\s*(\d+)\b", (topic_title or "").lower())
    if m:
        unit_num = int(m.group(1))

    if ap and unit_num:
        subj_core = re.sub(r"\bap\b", "", subject, flags=re.I).strip()
        queries = [
            f"AP {subj_core} Unit {unit_num} review",
            f"AP {subj_core} Unit {unit_num} full review",
            f"AP {subj_core} Unit {unit_num} exam review",
            f"AP {subj_core} Unit {unit_num} summary",
        ]
    else:
        queries = [
            f"{topic_title} unit review",
            f"{topic_title} chapter review",
            f"{topic_title} full review",
            f"{subject} {topic_title} review",
            f"{subject} {topic_title} summary",
        ]

    def strong_match(results):
        for v in results:
            title = (v.get("title") or "").lower()
            link = v.get("link") or ""
            if not link:
                continue
            if ap and unit_num is not None:
                if re.search(rf"\bunit\s*{unit_num}\b", title) and "review" in title:
                    return link
            else:
                if ("review" in title) and ("unit" in title or "chapter" in title or "full" in title or "complete" in title):
                    return link
        return ""

    last_seen = ""
    for q in queries:
        try:
            res = VideosSearch(q, limit=8).result().get("result", [])
        except Exception:
            continue
        link = strong_match(res)
        if link:
            return link
        if res:
            last_seen = res[0].get("link") or last_seen

    if last_seen:
        return last_seen
    try:
        res = VideosSearch(f"{subject} {topic_title}", limit=1).result().get("result", [])
        return res[0]["link"] if res else ""
    except Exception:
        return ""

def search_extra_videos(subject):
    from youtubesearchpython import VideosSearch
    try:
        res = VideosSearch(f"{subject} review", limit=6).result().get("result", [])
        return [{"title": v.get("title"), "link": v.get("link")} for v in res][:6]
    except Exception:
        return []

# ---------------------------------- ROUTES ---------------------------------- #

@app.route("/", methods=["GET", "POST"])
def home():
    output = None
    if request.method == "POST":
        subject = request.form.get("subject", "").strip()
        past_titles = request.form.getlist("past_topics")
        total_done = len(past_titles)

        if past_titles:
            learned = "\n".join([f"- {t}" for t in past_titles])
            prompt = f"""
You are designing a course on "{subject}".
They already learned:
{learned}

Now generate 3 new beginner-friendly lessons by unit or key topic.
If AP, use official AP unit names (e.g., "Unit 3: Cultural Patterns and Processes").
Return ONLY JSON array: [{{"title":"...","description":"..."}}, ...]
"""
        else:
            prompt = f"""
Create the first 3 beginner lessons for "{subject}" (by unit or key topic).
If AP, use official AP unit names.
Return ONLY JSON array: [{{"title":"...","description":"..."}}, ...]
"""

        # Use your known‑working model
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}]
        )
        try:
            new_lessons = parse_json_strict(resp.choices[0].message.content, want="array")
        except Exception:
            return render_template("index.html", output={
                "subject": subject,
                "topics": [],
                "error": "❌ GPT failed to generate lessons. Please try again."
            })

        # keep old lessons if continuing
        structured = []
        for i, pt in enumerate(past_titles):
            description = request.form.getlist("past_descriptions")[i] if request.form.getlist("past_descriptions") else ""
            video = request.form.getlist("past_videos")[i] if request.form.getlist("past_videos") else ""
            quiz_json = request.form.getlist("past_quizzes")[i] if request.form.getlist("past_quizzes") else "[]"
            try:
                quiz = json.loads(quiz_json)
            except:
                quiz = []
            structured.append({
                "title": pt if pt.lower().startswith("lesson") else f"Lesson {i+1}: {pt}",
                "description": description,
                "video": video,
                "quiz": quiz
            })

        # new lessons + videos + mini‑quizzes
        for i, lesson in enumerate(new_lessons):
            n = total_done + i + 1
            title = lesson.get("title", f"Lesson {n}")
            description = lesson.get("description", "")
            video = search_youtube_precise(subject, title)

            quiz_prompt = f"""
Create 2 specific multiple choice questions for "{title}" ({subject}).
Return ONLY JSON array:
[
  {{
    "question": "...",
    "options": ["A) ...","B) ...","C) ...","D) ..."],
    "answer": "B"
  }},
  {{
    "question": "...",
    "options": ["A) ...","B) ...","C) ...","D) ..."],
    "answer": "C"
  }}
]
"""
            quiz_resp = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":quiz_prompt}]
            )
            try:
                quiz = parse_json_strict(quiz_resp.choices[0].message.content, want="array")
            except:
                quiz = []

            structured.append({
                "title": f"Lesson {n}: {title}" if not title.lower().startswith("lesson") else title,
                "description": description,
                "video": video,
                "quiz": quiz
            })

        all_quiz = []
        for item in structured:
            if isinstance(item.get("quiz"), list):
                all_quiz.extend(item["quiz"])

        output = {
            "subject": subject,
            "topics": structured,
            "quiz": all_quiz,
            "videos": search_extra_videos(subject),
            "channel": f"CrashCourse / Khan Academy (auto-picked)",
            "plan": f"This course contains {len(structured)} lessons."
        }

    return render_template("index.html", output=output)

@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    data = request.get_json(force=True, silent=True) or {}
    subject = data.get("subject", "")
    all_titles = data.get("topics", [])

    prompt = f"""Generate 10 specific multiple choice questions for "{subject}" covering:
""" + "\n".join(f"- {t}" for t in all_titles) + """
Return ONLY JSON array:
[
  {
    "question": "...",
    "options": ["A) ...","B) ...","C) ...","D) ..."],
    "answer": "C"
  }
  ...
]
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user","content":prompt}]
    )
    try:
        quiz = parse_json_strict(response.choices[0].message.content, want="array")
        return jsonify({"quiz": quiz})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["GET"])
def upload():
    return render_template("upload.html")

@app.route("/generate_<task>_from_doc", methods=["POST"])
def generate_from_doc(task):
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400

    filename = secure_filename(file.filename)
    ext = filename.split('.')[-1].lower()
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    if ext == "pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(filepath)
            text = "".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            text = ""
    else:
        try:
            with open(filepath, "rb") as f:
                text = f.read().decode("utf-8", errors="ignore")
        except Exception:
            text = ""

    if not text.strip():
        return jsonify({"error": "File is empty or unreadable"}), 400

    try:
        if task == "flashcards":
            prompt = f"""Create 12 concise flashcards from the content below.
Return ONLY JSON array: [{{"term":"...","definition":"..."}}, ...]
CONTENT:
{text[:3500]}"""
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role":"user","content":prompt}]
            )
            flashcards = parse_json_strict(response.choices[0].message.content, want="array")
            return jsonify({"flashcards": flashcards})

        elif task == "quiz":
            prompt = f"""Create 10 specific multiple-choice questions from the content below.
Return ONLY JSON array:
[
  {{
    "question":"...",
    "options":["A) ...","B) ...","C) ...","D) ..."],
    "answer":"B"
  }},
  ...
]
CONTENT:
{text[:3500]}"""
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}]
            )
            quiz = parse_json_strict(response.choices[0].message.content, want="array")
            return jsonify({"quiz": quiz})

        elif task == "studyguide":
            if len(text.split()) < 150:
                return jsonify({"error": "Study guide too short to extract lessons."}), 400
            prompt = f"""Extract 3 major topics with 1‑sentence summaries from the content below.
Return ONLY JSON array: [{{"title":"...","summary":"..."}}, ...]
CONTENT:
{text[:3500]}"""
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role":"user","content":prompt}]
            )
            topics = parse_json_strict(response.choices[0].message.content, want="array")
            for t in topics:
                t["video"] = search_youtube_precise("", t.get("title",""))
            return jsonify({"studyguide": topics})

        return jsonify({"error": "Invalid task"}), 400

    except Exception as e:
        return jsonify({"error": f"Failed to generate: {str(e)}"}), 500

@app.route("/flashcards/<subject>")
def flashcards_page(subject):
    return render_template("flashcards.html", subject=subject)

@app.route('/generate_flashcard_quiz', methods=['POST'])
def generate_flashcard_quiz():
    data = request.get_json(force=True, silent=True) or {}
    subject = data.get('subject', '')
    count = int(data.get('questionCount', 5))
    difficulty = data.get('difficulty', 'Medium')

    prompt = f"""
Generate {count} multiple-choice questions about "{subject}" at {difficulty} difficulty.
Return ONLY JSON array:
[
  {{
    "question":"...",
    "options":["A) ...","B) ...","C) ...","D) ..."],
    "answer":"A"
  }},
  ...
]
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role":"system","content":"You are a precise quiz generator."},
                {"role":"user","content":prompt}
            ]
        )
        quiz = parse_json_strict(response.choices[0].message.content, want="array")
        return jsonify({"success": True, "quiz": quiz})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route('/generate_flashcards_from_subject', methods=['POST'])
def generate_flashcards_from_subject():
    data = request.get_json(force=True, silent=True) or {}
    subject = data.get("subject", "").strip()
    if not subject:
        return jsonify({"success": False, "error": "No subject provided"})

    prompt = f"""Generate 12 flashcards for "{subject}".
Return ONLY JSON array: [{{"term":"...","definition":"..."}}, ...]"""

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role":"user","content":prompt}]
        )
        flashcards = parse_json_strict(response.choices[0].message.content, want="array")
        return jsonify({"success": True, "flashcards": flashcards})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
