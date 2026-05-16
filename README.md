#  AI Object Detection System

A professional, and high-performance object detection application built with **YOLOv5** and **Streamlit**. This project provides real-time detection capabilities via webcam, as well as support for image and video file uploads with structured results.

---

## 🚀 Features

- **YOLOv5 Integration**: Powered by the latest Ultralytics implementation for superior accuracy and speed.
- **Web UI**: Interactive dashboard built with Streamlit for a seamless user experience.
- **Triple Mode Detection**:
  - 🎥 **Real-time Webcam**: Live object tracking with FPS monitoring.
  - 🖼️ **Image Upload**: Detailed analysis of static images with structured confidence reports.
  - 📽️ **Video Processing**: Frame-by-frame detection for uploaded video files.
- **Hardware Optimized**: Automatic GPU (CUDA) detection and FP16 half-precision support for maximum performance.
- **Interactive Thresholds**: Adjust confidence levels on the fly using the built-in sidebar.

---

## 📽️ Project Demo

> **[PLACEHOLDER: Replace this with your demo video link or gif]**

---

## 🛠️ Technical Stack

- **Language**: Python 3.8+
- **ML Framework**: [PyTorch](https://pytorch.org/) & [YOLOv5](https://github.com/ultralytics/yolov5)
- **UI Framework**: [Streamlit](https://streamlit.io/)
- **Computer Vision**: [OpenCV](https://opencv.org/) & [Pillow](https://python-pillow.org/)

---

## 📦 Installation & Setup

1. **Clone the repository**:
   ```powershell
   git clone https://github.com/NARMEEN-UROOJ/codealpha_objectDetection.git
   cd codealpha_objectDetection
   ```

2. **Create and activate a Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

3. **Install Dependencies**:
   ```powershell
   pip install torch torchvision ultralytics streamlit opencv-python pillow
   ```

4. **Run the Application**:
   ```powershell
   streamlit run app.py
   ```

---

## 📂 Project Structure

- `app.py`: The main Streamlit web application.
- `object_detector_v5.py`: Core logic for the YOLOv5 detector.
- `.gitignore`: Configured to exclude large model weights and environment files.

---

## 👤 Author

**Narmeen Urooj**
- GitHub: [@NARMEEN-UROOJ](https://github.com/NARMEEN-UROOJ)

