import gradio as gr
import cv2
import numpy as np
from ultralytics import YOLO
import os
from PIL import Image
import tempfile
import time

# Load model YOLOv8 dengan error handling
try:
    model = YOLO("banana.pt")
    print("Model berhasil dimuat!")
    print(f"Classes yang tersedia: {model.names}")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def detect_image(image):
    try:
        if image is None:
            return None, "Tidak ada gambar yang diupload"

        img_array = np.array(image)
        results = model(img_array)[0]
        annotated = results.plot()
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        detections = 0
        summary = "Tidak ada objek yang terdeteksi"

        if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
            detections = len(results.boxes)
            summary = f"Terdeteksi {detections} objek"
            try:
                classes = results.boxes.cls.cpu().numpy()
                class_names = [model.names.get(int(cls), f"Unknown_{int(cls)}") for cls in classes]
                class_counts = {name: class_names.count(name) for name in set(class_names)}
                summary += "\nDetail:\n" + "\n".join([f"- {k}: {v}" for k, v in class_counts.items()])
            except Exception as e:
                summary += f"\n(Error dalam detail klasifikasi: {str(e)})"

        return annotated_rgb, summary

    except Exception as e:
        return None, f"Error dalam deteksi gambar: {str(e)}"

def webcam_inference(image):
    try:
        if image is None:
            return None
        img_array = np.array(image) if isinstance(image, Image.Image) else image
        results = model(img_array)[0]
        annotated = results.plot()
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error dalam inference real-time: {e}")
        return np.array(image) if isinstance(image, Image.Image) else image

def get_detection_summary(image):
    try:
        if image is None:
            return "Tidak ada frame dari webcam"
        img_array = np.array(image) if isinstance(image, Image.Image) else image
        results = model(img_array)[0]

        detections = 0
        summary = "🔴 LIVE - Tidak ada objek terdeteksi"

        if hasattr(results, 'boxes') and results.boxes is not None and len(results.boxes) > 0:
            detections = len(results.boxes)
            summary = f"🔴 LIVE - Terdeteksi {detections} objek"
            try:
                classes = results.boxes.cls.cpu().numpy()
                class_names = [model.names.get(int(cls), f"Unknown_{int(cls)}") for cls in classes]
                class_counts = {name: class_names.count(name) for name in set(class_names)}
                summary += "\n📊 Detail Real-time:\n" + "\n".join([f"🍌 {k}: {v}" for k, v in class_counts.items()])
            except Exception as e:
                summary += f"\n(Error dalam detail: {str(e)})"

        current_time = time.strftime("%H:%M:%S")
        return summary + f"\n⏰ Update: {current_time}"

    except Exception as e:
        return f"Error dalam analisis: {str(e)}"

# CSS
css = """
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
}
h1 {
    text-align: center;
    color: #2E8B57;
    margin-bottom: 30px;
}
.live-indicator {
    background-color: #ff4444;
    color: white;
    padding: 5px 10px;
    border-radius: 15px;
    font-weight: bold;
    display: inline-block;
    margin-bottom: 10px;
}
"""

# Gradio Interface
with gr.Blocks(css=css, title="🍌 Deteksi Penyakit Daun Pisang") as app:
    gr.Markdown("# 🍌 Sistem Deteksi Penyakit Daun Pisang Real-time 🌿")

    with gr.Tabs():
        # Tab Gambar
        with gr.TabItem("📷 Upload Gambar"):
            gr.Markdown("### Upload gambar daun pisang untuk mendeteksi penyakit")
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(type="pil", label="Upload Gambar")
                    detect_btn = gr.Button("🔍 Deteksi Penyakit", variant="primary")
                with gr.Column():
                    image_output = gr.Image(label="Hasil Deteksi")
                    summary_output = gr.Textbox(label="Ringkasan Deteksi", lines=5)
            detect_btn.click(fn=detect_image, inputs=image_input, outputs=[image_output, summary_output])

        # Tab Webcam Live
        with gr.TabItem("📹 Real-time Webcam"):
            gr.Markdown("""
            ### 🔴 Deteksi Penyakit Daun Pisang via Webcam
            *Real-time bounding box tanpa tombol. Cocok untuk demo langsung!*
            """)
            with gr.Row():
                with gr.Column():
                    webcam_live = gr.Interface(
                        fn=webcam_inference,
                        inputs=gr.Image(sources="webcam", streaming=True, label="🔴 Kamera Live"),
                        outputs=gr.Image(label="📸 Deteksi Real-time"),
                        live=True,
                        allow_flagging="never",
                    )
                with gr.Column():
                    summary_interface = gr.Interface(
                        fn=get_detection_summary,
                        inputs=gr.Image(sources="webcam", streaming=True, label="🔄 Kamera Ringkasan"),
                        outputs=gr.Textbox(label="📈 Info Deteksi", lines=8),
                        live=True,
                        allow_flagging="never",
                    )

    gr.Markdown(
        """
        ---
        ### 🔧 Informasi Teknis
        - Model: YOLOv8 Custom (banana.pt)
        - Deteksi: Cordana, Fusarium, Healthy, Sigatoka
        - Format Gambar: JPG, PNG, WEBP

        ### 📹 Panduan
        - Gunakan webcam atau upload gambar daun
        - Pastikan pencahayaan cukup agar hasil akurat
        - Deteksi real-time bekerja otomatis (tidak perlu tombol)

        """
    )

if __name__ == "__main__":
    app.launch(share=True, debug=True, inbrowser=False)
