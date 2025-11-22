import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2
import os
import shutil


class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        self.last_prediction = None

        # ---------------------------
        # 🔥 Load model only once
        # ---------------------------
        model_path = "model/model.h5"
        if not hasattr(PredictionPipeline, "model"):
            PredictionPipeline.model = load_model(model_path)

        self.model = PredictionPipeline.model


    # ------------------------------------------------------------
    # 🔥 PERFECT Grad-CAM
    # ------------------------------------------------------------
    def generate_gradcam(self, layer_name="block5_conv3"):
        model = self.model

        # Load full image
        orig = cv2.imread(self.filename)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        oh, ow = orig.shape[:2]

        # Resize for model
        resized = cv2.resize(orig, (224, 224))
        x = np.expand_dims(resized / 255.0, axis=0)

        preds = model.predict(x)
        pred_idx = np.argmax(preds[0])

        grad_model = tf.keras.models.Model(
            [model.inputs],
            [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            loss = predictions[:, pred_idx]

        grads = tape.gradient(loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))

        cam = np.zeros(conv_outputs[0].shape[0:2], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * conv_outputs[0][:, :, i]

        cam = np.maximum(cam, 0)
        cam /= (cam.max() + 1e-8)

        heatmap = cv2.resize(cam, (ow, oh))

        # Mask background
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        heatmap *= (mask.astype("float32") / 255.0)
        heatmap /= (heatmap.max() + 1e-8)

        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        alpha = 0.45 if self.last_prediction == "Tumor" else 0.25
        blended = cv2.addWeighted(orig, 1 - alpha, heatmap_color, alpha, 0)

        # Save result safely
        os.makedirs("static", exist_ok=True)
        out_path = os.path.join("static", "gradcam_result.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        return out_path


    # ------------------------------------------------------------
    # 📌 MAIN PREDICTION
    # ------------------------------------------------------------
    def predict(self):
        model = self.model

        # Preprocess
        img = image.load_img(self.filename, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        confidence = float(np.max(preds)) * 100
        cls = np.argmax(preds)

        prediction = "Tumor" if cls == 1 else "Normal"
        self.last_prediction = prediction

        # Save original
        os.makedirs("static", exist_ok=True)
        orig_path = "static/original.jpg"
        shutil.copy(self.filename, orig_path)

        # Generate heatmap if tumor
        gradcam_path = None
        if prediction == "Tumor":
            gradcam_path = self.generate_gradcam()

        # EXTRA data for your report
        report_data = {
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "note": (
                "Possible kidney abnormality detected."
                if prediction == "Tumor"
                else "Kidney appears normal."
            ),
            "recommendation": (
                "Consult a radiologist."
                if prediction == "Tumor"
                else "Routine monitoring recommended."
            )
        }

        return [{
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "gradcam_path": gradcam_path,
            "original_image_path": orig_path,
            "report": report_data
        }]
