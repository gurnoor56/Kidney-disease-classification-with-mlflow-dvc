# ---------------------------------------------------
# app.py — NOOR AI (Final Clean Version)
# ---------------------------------------------------

from flask import (
    Flask, request, jsonify, render_template, send_from_directory,
    redirect, url_for, session, flash
)
import os
import json
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from flask_cors import CORS
from datetime import datetime
from pathlib import Path

# PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

# project imports
from KidneyClassification.utils.common import decodeImage
from KidneyClassification.pipeline.prediction import PredictionPipeline


# ---------------------------------------------------
# Flask Setup
# ---------------------------------------------------
app = Flask(__name__)
CORS(app)

app.secret_key = os.environ.get("SECRET_KEY", "change-me")
USERS_FILE = "users.json"


# ---------------------------------------------------
# User storage
# ---------------------------------------------------
def ensure_default_users():
    if not os.path.exists(USERS_FILE):
        users = {
            "admin": {
                "password": generate_password_hash("admin123"),
                "display_name": "Administrator"
            }
        }
        with open(USERS_FILE, "w") as f:
            json.dump(users, f, indent=2)


def load_users():
    if not os.path.exists(USERS_FILE):
        ensure_default_users()
    return json.load(open(USERS_FILE))


def save_users(users: dict):
    json.dump(users, open(USERS_FILE, "w"), indent=2)


ensure_default_users()


# ---------------------------------------------------
# Auth decorator
# ---------------------------------------------------
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            flash("Please login first.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ---------------------------------------------------
# Prediction Pipeline Wrapper
# ---------------------------------------------------
latest_result = {}


class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


clApp = ClientApp()


# ---------------------------------------------------
# Routes
# ---------------------------------------------------

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]
        display = request.form.get("display_name", username)

        users = load_users()
        if username in users:
            flash("Username already exists.", "danger")
            return redirect("/register")

        users[username] = {
            "password": generate_password_hash(password),
            "display_name": display
        }
        save_users(users)
        flash("Registration successful!", "success")
        return redirect("/login")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"].strip()
        password = request.form["password"]

        users = load_users()
        user = users.get(username)

        if not user or not check_password_hash(user["password"], password):
            flash("Invalid credentials", "danger")
            return redirect("/login")

        session["username"] = username
        session["display_name"] = user["display_name"]
        return redirect("/")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
@login_required
def home():
    return render_template("index.html")


# ---------------------------------------------------
# Predict Route
# ---------------------------------------------------
@app.route("/predict", methods=["POST"])
@login_required
def predictRoute():
    global latest_result

    try:
        image = request.json.get("image")
        decodeImage(image, clApp.filename)

        result = clApp.classifier.predict()[0]

        latest_result = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "gradcam_path": result["gradcam_path"],
            "original_image_path": result["original_image_path"],
        }

        return jsonify({"status": "success", "result": latest_result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# ---------------------------------------------------
# Heatmap Page
# ---------------------------------------------------
@app.route("/heatmap")
@login_required
def heatmap_page():
    if not latest_result:
        return render_template("heatmap.html", data={"prediction": None})

    return render_template("heatmap.html", data=latest_result)


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory("static", filename)


# ---------------------------------------------------
# PDF Report Generator
# ---------------------------------------------------
def _ensure_reports():
    Path("static/reports").mkdir(parents=True, exist_ok=True)
    return Path("static/reports")


def generate_report_pdf(display_name, result):
    """
    Always save as:
        Kidney_Report_<timestamp>.pdf
    """
    reports_dir = _ensure_reports()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = f"Kidney_Report_{timestamp}.pdf"
    path = reports_dir / filename

    c = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    margin = 40
    y = height - margin

    c.setFont("Helvetica-Bold", 20)
    c.drawString(margin, y, "NOOR AI - Kidney Report")
    y -= 35

    c.setFont("Helvetica", 11)
    c.drawString(margin, y, f"User: {display_name}")
    y -= 15
    c.drawString(margin, y, f"Prediction: {result['prediction']}")
    y -= 15
    c.drawString(margin, y, f"Confidence: {result['confidence']}")
    y -= 20

    c.line(margin, y, width - margin, y)
    y -= 20

    # Images
    def draw_img(img_path, caption):
        nonlocal y
        if not img_path:
            return
        try:
            img = ImageReader(img_path)
            iw, ih = img.getSize()
            scale = min(450 / iw, 250 / ih)
            w = iw * scale
            h = ih * scale

            if y - h < 50:
                c.showPage()
                y = height - margin

            c.setFont("Helvetica-Bold", 12)
            c.drawString(margin, y, caption)
            y -= 15
            c.drawImage(img, margin, y - h, width=w, height=h)
            y -= h + 20
        except:
            pass

    draw_img(result["original_image_path"], "Original Image")
    draw_img(result["gradcam_path"], "Heatmap (GradCAM)")

    c.setFont("Helvetica-Oblique", 9)
    c.drawString(margin, 20, "Generated by NOOR AI")

    c.save()
    return str(path)


@app.route("/download_report")
@login_required
def download_report():
    if not latest_result:
        flash("No prediction yet!", "warning")
        return redirect("/")

    pdf_path = generate_report_pdf(session["display_name"], latest_result)

    return send_from_directory(
        "static/reports",
        os.path.basename(pdf_path),
        as_attachment=True
    )


# ---------------------------------------------------
# Run App
# ---------------------------------------------------
if __name__ == "__main__":
    print("\nRunning NOOR AI on http://127.0.0.1:8080\n")
    app.run(host="0.0.0.0", port=8080)
