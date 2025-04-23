## 🚀 Workplace Activity Recognition System

Automatically recognize and classify workplace activities such as *Working*, *Idle*, and *Empty Chair* using computer vision and YOLO object detection models (YOLOv8 & YOLOv11). Designed to respect employee privacy while offering valuable productivity insights.

---

## 📌 Project Steps

### 1. **Business Understanding**
- **Problem:** Traditional employee monitoring is either inefficient or intrusive.
- **Goal:** Develop a privacy-preserving, real-time activity recognition system to optimize workplace productivity.

### 2. **Data Collection & Annotation**
- Collected **annotated images and videos** showing workplace scenarios (Idle, Working, Empty Chair).
- **Annotation Tool:** [Roboflow](https://roboflow.com)
  - Applied bounding boxes
  - Preprocessing: resizing, augmentation (rotation, scaling), normalization
- **Dataset Size:**
  - YOLOv8: 2,067 images
  - YOLOv11: 2,112 images

### 3. **System Setup**
- **Environment:** Google Colab (NVIDIA T4 GPU)
- **Languages & Tools:** Python, PyTorch, OpenCV, Numpy, Roboflow, YOLOv8/YOLOv11, Streamlit
- **Storage:** Google Drive integration

### 4. **Model Development**
- **Model Types:** YOLOv8s, YOLOv11s, YOLOv8n, YOLOv11n
- **Training Strategy:**
  - Format: YOLOv8 & YOLOv11
  - Transfer learning enabled
  - Batch Size: 16, Epochs: 10
- **Evaluation Metrics:**
  - Precision, Recall, F1-Score, mAP (50 & 50-95)

### 5. **Model Evaluation**
- **Best Model:** YOLOv8s
  - Precision: 0.925
  - Recall: 0.922
  - mAP50: 0.958
  - mAP50-95: 0.811
- Best performance especially for “Empty Chair” class (Precision: 0.96, Recall: 0.95)

### 6. **Model Deployment**
- **Deployment Platform:** [Streamlit](https://streamlit.io)
- **Features:**
  - Upload image/video
  - Real-time activity recognition
  - Confidence score visualization

### 7. **Challenges Faced**
- Data diversity and annotation accuracy
- Real-time inference vs. accuracy trade-off
- Privacy and ethical data handling
- Hardware limitations (local deployment)
- Generalization across different environments

### 8. **Future Scope**
- Add more activity classes (e.g., Meetings, Breaks)
- Real-time analytics dashboard
- Integration with workplace tools
- Scalable to larger/multi-site workplaces
- Privacy-preserving AI techniques

---



