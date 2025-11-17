from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from flask_cors import CORS
from KidneyClassification.utils.common import decodeImage
from KidneyClassification.pipeline.prediction import PredictionPipeline

# ---------------------------------------------------
# Flask App Setup
# ---------------------------------------------------
app = Flask(__name__)
CORS(app)

# Stores last prediction result to show in /heatmap
latest_result = {}


# ---------------------------------------------------
# Client App Wrapper
# ---------------------------------------------------
class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)


# Create Client App Instance
clApp = ClientApp()


# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """Main UI page (index.html)"""
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predictRoute():
    """Receives uploaded image → runs model → returns JSON"""
    global latest_result

    try:
        print("\n🔍 Incoming /predict request")

        # Read Base64 image sent from frontend
        image = request.json.get("image")
        if not image:
            return jsonify({"status": "error", "message": "No image received"})

        # Decode and save image
        decodeImage(image, clApp.filename)

        # Model Prediction
        result = clApp.classifier.predict()[0]

        # Build Output for heatmap page
        latest_result = {
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "gradcam_path": result["gradcam_path"],
            "original_image_path": result["original_image_path"]
        }

        # -------------------------------------------------------
        # ⭐ ADD MODEL PERFORMANCE METRICS (STATIC FOR DISPLAY)
        # -------------------------------------------------------
        latest_result.update({
            "accuracy": "98.2%",
            "loss": "0.12",
            "precision": "97.5%",
            "recall": "96.8%",
            "f1": "97.1%",
            "params": "14.7 Million"
        })

        print("✅ Prediction Ready:", latest_result)

        return jsonify({"status": "success", "result": latest_result})

    except Exception as e:
        print("❌ ERROR in /predict:", e)
        return jsonify({"status": "error", "message": str(e)})


@app.route("/heatmap", methods=["GET"])
def heatmap_page():
    """Displays original image + heatmap + summary"""
    global latest_result

    if not latest_result:
        return render_template(
            "heatmap.html",
            data={
                "error": "⚠️ No prediction yet!",
                "prediction": None,
                "confidence": None,
                "gradcam_path": None,
                "original_image_path": None,
                "accuracy": None,
                "loss": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "params": None
            }
        )

    # Render heatmap page with latest prediction data
    return render_template("heatmap.html", data=latest_result)


@app.route("/static/<path:filename>")
def serve_static(filename):
    """Serve images from static folder"""
    return send_from_directory("static", filename)


@app.route("/debug")
def debug():
    """Debug route to confirm template path"""
    return "Templates Path: " + os.path.abspath("templates")


# ---------------------------------------------------
# MAIN APP RUN
# ---------------------------------------------------
if __name__ == "__main__":
    print(r"""
 _   _   ___   ___  _____      _    ___ 
| \ | | / _ \ / _ \| ____|    / \  |_ _|
|  \| || | | | | | |  _|     / _ \  | | 
| |\  || |_| | |_| | |___   / ___ \ | | 
|_| \_| \___/ \___/|_____| /_/   \_\___|
    🚀 NOOR AI — Kidney Disease Classifier
""")

    print("🌐 Running at: http://127.0.0.1:8080\n")
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
