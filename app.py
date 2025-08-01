from flask import Flask, render_template, request
from youtubesearchpython import VideosSearch
import json
import os
from dotenv import load_dotenv
load_dotenv(override=True)
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
load_dotenv()

print("KEY LOADED:", os.getenv("OPENAI_API_KEY"))


app = Flask(__name__)
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def search_youtube(topic):
    from youtubesearchpython import VideosSearch

    # Refined query: encourage longer/better videos
    search_query = f"{topic} full lesson OR review OR summary"

    results = VideosSearch(search_query, limit=5).result()["result"]

    # Define preferred filters
    preferred_keywords = ["unit review", "unit summary", "unit", "chapter", "full", "complete", "highschool"]
    excluded_keywords = ["intro", "basic", "easy", "kindergarten", "middle school"]

    for video in results:
        title = video["title"].lower()
        if any(p in title for p in preferred_keywords) and not any(e in title for e in excluded_keywords):
            return video["link"]

    # fallback
    return results[0]["link"] if results else ""


def search_extra_videos(subject):
    results = VideosSearch(f"{subject} simplified", limit=3).result()["result"]
    return [{"title": v["title"], "link": v["link"]} for v in results]

@app.route("/", methods=["GET", "POST"])
def home():
    output = None
    if request.method == "POST":
        subject = request.form.get("subject", "")
        past_titles = request.form.getlist("past_topics")
        total_lessons_done = len(past_titles)

        if past_titles:
            learned = "\n".join([f"- {t}" for t in past_titles])
            prompt = f"""
You are designing a course on \"{subject}\".
The user has already learned:
{learned}

Now generate 3 new structured beginner-level lessons.
Each lesson should be a dictionary like this:
{{"title": "...", "description": "..."}}
Return a Python list like this:
[
  {{"title": "...", "description": "..."}},
  ...
]"""
        else:
            prompt = f"""
Create the first 3 beginner lessons for a course on \"{subject}\".
Each lesson should be a dictionary like this:
{{"title": "...", "description": "..."}}
Return a Python list like this:
[
  {{"title": "...", "description": "..."}},
  ...
]"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            new_lessons = eval(response.choices[0].message.content.strip())
            if not isinstance(new_lessons, list):
                raise ValueError("Bad format")
        except:
            return render_template("index.html", output={
                "subject": subject,
                "topics": [],
                "error": "❌ GPT failed to generate lessons. Please try again."
            })

        structured_past = []

        for i, pt in enumerate(past_titles):
            description = request.form.getlist("past_descriptions")[i] if request.form.getlist("past_descriptions") else ""
            video = request.form.getlist("past_videos")[i] if request.form.getlist("past_videos") else ""
            quiz_json = request.form.getlist("past_quizzes")[i] if request.form.getlist("past_quizzes") else "[]"
            try:
                quiz = json.loads(quiz_json)
            except:
                quiz = []
            structured_past.append({
                "title": pt if pt.startswith("Lesson") else f"Lesson {i + 1}: {pt}",
                "description": description,
                "video": video,
                "quiz": quiz
            })

        for i, lesson in enumerate(new_lessons):
            lesson_number = total_lessons_done + i + 1
            title = lesson["title"]
            description = lesson["description"]
            video = search_youtube(f"{subject} - {title}")

            quiz_prompt = f"""
Create 2 multiple choice questions for a lesson titled: \"{title}\".
Format as a Python list like this:
[
  {{
    'question': '...',
    'options': ['A) ...', 'B) ...', 'C) ...', 'D) ...'],
    'answer': 'B'
  }},
  ...
]
"""
            quiz_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": quiz_prompt}]
            )

            try:
                quiz = eval(quiz_response.choices[0].message.content.strip())
            except:
                quiz = []

            structured_past.append({
                "title": f"Lesson {lesson_number}: {title}" if not title.startswith("Lesson") else title,
                "description": description,
                "video": video,
                "quiz": quiz
            })

        all_quiz = []
        for item in structured_past:
            if "quiz" in item:
                all_quiz.extend(item["quiz"])

        extra_videos = search_extra_videos(subject)

        output = {
            "subject": subject,
            "topics": structured_past,
            "quiz": all_quiz,
            "videos": extra_videos,
            "channel": f"CrashCourse or Khan Academy for {subject}",
            "plan": f"This course contains {len(structured_past)} lessons to help you master {subject}."
        }

    return render_template("index.html", output=output)
from flask import jsonify

@app.route("/generate_quiz", methods=["POST"])
def generate_quiz():
    from openai import OpenAI
    import os
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    subject = request.json.get("subject", "")
    all_titles = request.json.get("topics", [])

    prompt = f"""Generate 10 multiple choice questions for a course on "{subject}" that covers the following lessons:\n"""
    for t in all_titles:
        prompt += f"- {t}\n"
    prompt += """
Each question should be extremely specific and challenging that will test thier full understnading of the subject, if math, give questions that take time to solve and are specific. Format like this:
[
  {
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "C"
  },
  ...
]
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        quiz = eval(response.choices[0].message.content.strip())
        return jsonify({"quiz": quiz})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/upload', methods=['GET', 'POST'])
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
        import PyPDF2
        reader = PyPDF2.PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
    else:
        text = file.read().decode("utf-8", errors="ignore")

    if not text.strip():
        return jsonify({"error": "File is empty or unreadable"}), 400

    try:
        if task == "flashcards":
            prompt = f"Create 10 flashcards in this JSON format:\n[\n{{\"term\": \"...\", \"definition\": \"...\"}},\n...]\n\nContent:\n{text[:3000]}"
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
            raw = response.choices[0].message.content.strip()
            print("\n📘 RAW FLASHCARDS OUTPUT:\n", raw)

            flashcards = json.loads(raw)
            return jsonify({"flashcards": flashcards})

        elif task == "quiz":
            prompt = f"""Create 10 specific and challenging multiple-choice questions in JSON format like:
[
  {{
    "question": "...",
    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
    "answer": "B"
  }},
  ...
]
Content:\n{text[:3000]}"""

            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
            raw = response.choices[0].message.content.strip()
            print("\n🟦 RAW QUIZ OUTPUT:\n", raw)

            import re
            json_match = re.search(r"\[.*\]", raw, re.DOTALL)
            if not json_match:
                return jsonify({"error": "❌ Could not extract JSON from GPT output."}), 500

            try:
                quiz = json.loads(json_match.group(0))
            except Exception as e:
                return jsonify({"error": f"❌ Failed to parse quiz JSON: {str(e)}"}), 500

            return jsonify({"quiz": quiz})

        elif task == "studyguide":
            if len(text.split()) < 150:
                return jsonify({"error": "Study guide too short to extract lessons."}), 400
            prompt = f"Extract 3 major topics from the following study guide and find 1 useful YouTube video link for each:\n\n{text[:3000]}"
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
            return jsonify({"studyguide": response.choices[0].message.content.strip()})

        return jsonify({"error": "Invalid task"}), 400

    except Exception as e:
        return jsonify({"error": f"Failed to generate: {str(e)}"}), 500

from flask import request, jsonify
@app.route("/flashcards/<subject>")
def flashcards_page(subject):
    return render_template("flashcards.html", subject=subject)

@app.route('/generate_flashcard_quiz', methods=['POST'])
def generate_flashcard_quiz():
    data = request.json
    subject = data.get('subject')
    count = data.get('questionCount', 5)
    difficulty = data.get('difficulty', 'Medium')

    prompt = f"""
    Generate {count} specific multiple choice quiz questions about {subject}.
    Difficulty level: {difficulty}.
    Each question should be JSON formatted like this:
    {{
      "question": "...",
      "options": ["A) ...", "B) ...", "C) ...", "D) ..."],
      "answer": "A"
    }}
    Return as a JSON array.
    """

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a quiz generator."},
            {"role": "user", "content": prompt}
        ]
    )

    try:
        raw = response.choices[0].message.content.strip()
        quiz = json.loads(raw)
        return jsonify({"success": True, "quiz": quiz})
    except Exception as e:
        return jsonify({"success": False, "error": str(e), "raw": raw})
@app.route('/generate_flashcards_from_subject', methods=['POST'])
def generate_flashcards_from_subject():
    subject = request.json.get("subject", "")
    print("SUBJECT RECEIVED:", subject)

    if not subject:
        return jsonify({"success": False, "error": "No subject provided"})

    prompt = f"Generate 10 flashcards for the subject: {subject}. Format as JSON: " \
             f"[{{\"term\": \"...\", \"definition\": \"...\"}}, ...]"

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response.choices[0].message.content.strip()
        print("RAW OPENAI RESPONSE:", raw)  # 👈 THIS IS KEY

        flashcards = json.loads(raw)
        return jsonify({"success": True, "flashcards": flashcards})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})





if __name__ == "__main__":
    app.run(debug=True)
