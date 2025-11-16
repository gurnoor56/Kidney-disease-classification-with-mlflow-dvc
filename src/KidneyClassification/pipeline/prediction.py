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


    # ------------------------------------------------------------
    # 🔥  Perfect Tumor-Detecting Grad-CAM (Latest Version)
    # ------------------------------------------------------------
    def generate_gradcam(self, model, layer_name="block5_conv3"):
        """Generate a perfectly aligned Grad-CAM + background masking."""

        # Load original full-resolution image
        orig = cv2.imread(self.filename)
        orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = orig.shape[:2]

        # Resize to model input (224x224)
        resized = cv2.resize(orig, (224, 224))
        x = np.expand_dims(resized / 255.0, axis=0)

        preds = model.predict(x)
        pred_index = np.argmax(preds[0])

        # Gradient model
        grad_model = tf.keras.models.Model(
            [model.inputs], 
            [model.get_layer(layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(x)
            loss = predictions[:, pred_index]

        grads = tape.gradient(loss, conv_outputs)[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))

        cam = np.zeros(conv_outputs[0].shape[0:2], dtype=np.float32)

        # Weighted sum of channels
        for i, w in enumerate(weights):
            cam += w * conv_outputs[0][:, :, i]

        # Normalize CAM
        cam = np.maximum(cam, 0)
        cam /= (cam.max() + 1e-8)

        # Resize CAM to original image size
        heatmap = cv2.resize(cam, (orig_w, orig_h))

        # ------------------------------------
        # ⭐ REMOVE BLACK BACKGROUND PROPERLY
        # ------------------------------------
        gray = cv2.cvtColor(orig, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        heatmap = heatmap * (mask.astype("float32") / 255.0)
        heatmap /= (heatmap.max() + 1e-8)

        # Convert heatmap to color
        heatmap_color = cv2.applyColorMap(
            np.uint8(255 * heatmap), cv2.COLORMAP_JET
        )
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Transparency control
        alpha = 0.45 if self.last_prediction == "Tumor" else 0.25

        blended = cv2.addWeighted(orig, 1 - alpha, heatmap_color, alpha, 0)

        # Save result
        os.makedirs("static", exist_ok=True)
        gradcam_path = os.path.join("static", "gradcam_result.jpg")
        cv2.imwrite(gradcam_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

        return gradcam_path


    # ------------------------------------------------------------
    # 🔍 PREDICTION + AUTO-HANDLING NORMAL/TUMOR
    # ------------------------------------------------------------
    def predict(self):
        """Predict Normal/Tumor + save heatmap + copy original image."""
        
        model_path = "artifacts/training/model.h5"
        model = load_model(model_path)

        # Preprocess
        img = image.load_img(self.filename, target_size=(224, 224))
        img = image.img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        preds = model.predict(img)
        confidence = float(np.max(preds)) * 100
        pred_class = np.argmax(preds)

        prediction = "Tumor" if pred_class == 1 else "Normal"
        self.last_prediction = prediction

        # --------------------------------------------------------
        # SAVE ORIGINAL IMAGE IN static/
        # --------------------------------------------------------
        os.makedirs("static", exist_ok=True)
        original_path = os.path.join("static", "original.jpg")
        shutil.copy(self.filename, original_path)

        # --------------------------------------------------------
        # ONLY generate heatmap if tumor
        # --------------------------------------------------------
        gradcam_path = None

        if prediction == "Tumor":
            gradcam_path = self.generate_gradcam(model)

        # Final output
        return [{
            "prediction": prediction,
            "confidence": f"{confidence:.2f}%",
            "gradcam_path": gradcam_path,
            "original_image_path": "static/original.jpg"
        }]
