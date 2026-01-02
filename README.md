# 🤟 American Sign Language (ASL) Recognition System

A **real-time American Sign Language (ASL) alphabet recognition system** built using **MediaPipe hand landmarks** and **Deep Learning**.
The system captures hand gestures via webcam, extracts 3D landmark features, trains a neural network model, and predicts ASL alphabets in real time.

---

## 📌 Description

Communication between hearing-impaired individuals and non-signers can be challenging.
This project aims to reduce that gap by providing an **automated ASL recognition system** using computer vision and machine learning techniques.

Instead of using raw images, the system leverages **hand landmark-based features**, making it lightweight, efficient, and suitable for real-time execution.

---

## ⚙️ Workflow

1. **Hand Gesture Capture**

   * Webcam captures live video input.
2. **Hand Landmark Detection**

   * MediaPipe detects 21 hand landmarks.
3. **Feature Extraction**

   * Each landmark provides (x, y, z) coordinates → 63 features.
4. **Model Training**

   * A neural network is trained using landmark features.
5. **Real-time Prediction**

   * The trained model predicts ASL alphabets live on screen.

---

## 🛠 Tech Stack

* **Programming Language:** Python
* **Computer Vision:** OpenCV, MediaPipe
* **Deep Learning:** TensorFlow / Keras
* **Data Processing:** NumPy
* **Machine Learning Utilities:** Scikit-learn

---

## 📂 Project Structure

```
ASL/
│── asl_dataset/          # Collected landmark CSV files
│   ├── A/
│   ├── B/
│   └── C/
│── venv/                 # Virtual environment
│── collection.py         # Dataset collection script
│── train.py              # Model training script
│── predict.py            # Real-time prediction script
│── asl_model.h5          # Trained model
│── README.md
```

---

## 🚀 Installation

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/ASL-Recognition.git
cd ASL-Recognition
```

### 2️⃣ Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install dependencies

```bash
pip install tensorflow opencv-python mediapipe numpy scikit-learn
```

---

## ▶️ Usage

### 🔹 Collect Dataset

```bash
python collection.py
```

* Press **S** → Save gesture
* Press **N** → Move to next letter
* Press **Q** → Quit

---

### 🔹 Train the Model

```bash
python train.py
```

---

### 🔹 Run Real-time Prediction

```bash
python predict.py
```

* Show ASL gesture in front of webcam
* Predicted letter appears on screen
* Press **Q** to exit

---

## 📊 Model Details

* **Input Features:** 63 (21 landmarks × 3 coordinates)
* **Architecture:**

  * Dense (128) → ReLU
  * Dense (64) → ReLU
  * Dense (Softmax)
* **Optimizer:** Adam
* **Loss Function:** Sparse Categorical Crossentropy

---

## 📈 Results

* Real-time ASL alphabet recognition achieved
* Accurate predictions for trained gestures
* Low latency and smooth webcam performance

---

## ⚠️ Limitations

* Supports limited alphabets (A, B, C)
* Performance depends on lighting conditions
* Background noise may affect detection accuracy

---

## 🔮 Future Scope

* Extend support to full ASL alphabet (A–Z)
* Word and sentence formation
* Text-to-speech integration
* Improve accuracy using LSTM / CNN-LSTM models
* Web deployment using Streamlit or Flask

---

## 👩‍💻 Author

**Mahek**
AI & Data Science Student

---
