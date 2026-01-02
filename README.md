
```markdown
# 🤟 American Sign Language (ASL) Recognition System

This project presents a **real-time American Sign Language (ASL) recognition system** using **MediaPipe hand landmarks** and **Deep Learning**.  
The system detects hand gestures through a webcam, extracts landmark features, trains a neural network model, and predicts ASL alphabets in real time.

---

## 📌 Project Overview
Communication barriers exist between hearing-impaired individuals and the general population.  
This project aims to reduce this gap by enabling **automatic recognition of ASL hand gestures** using computer vision and machine learning techniques.

---

## ✨ Features
- 🎥 Real-time hand gesture detection
- ✋ MediaPipe-based hand landmark extraction (21 landmarks)
- 🧠 Deep Learning model for classification
- 🔤 ASL alphabet recognition
- ⚡ Fast and lightweight execution
- 💻 Webcam-based live prediction

---

## 🗂 Project Structure
```

ASL/
│── asl_dataset/            # Dataset (CSV landmark files)
│   ├── A/
│   ├── B/
│   └── C/
│── venv/                   # Virtual environment
│── collection.py           # Dataset collection
│── train.py                # Model training
│── predict.py              # Real-time prediction
│── asl_model.h5             # Trained model
│── README.md

````

---

## 🛠 Technologies Used
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Scikit-learn

---

## ⚙️ System Workflow

### 1️⃣ Data Collection
- Webcam captures hand gestures
- MediaPipe extracts **21 hand landmarks**
- Each landmark has **x, y, z coordinates**
- Data saved as CSV files (63 features per sample)

---

### 2️⃣ Model Training
- CSV data is loaded and preprocessed
- Labels are encoded numerically
- Neural network is trained using Dense layers
- Trained model saved as `asl_model.h5`

---

### 3️⃣ Real-time Prediction
- Webcam input processed frame-by-frame
- Hand landmarks extracted
- Model predicts ASL alphabet
- Output displayed live on screen

---

## 🚀 Installation & Setup

### 🔹 Step 1: Create Virtual Environment
```bash
python -m venv venv
venv\Scripts\activate
````

### 🔹 Step 2: Install Dependencies

```bash
pip install tensorflow opencv-python mediapipe numpy scikit-learn
```

---

## ▶️ How to Run the Project

### 📌 Collect Dataset

```bash
python collection.py
```

* Press **S** → Save hand gesture
* Press **N** → Next letter
* Press **Q** → Quit

---

### 📌 Train the Model

```bash
python train.py
```

---

### 📌 Run Real-time Prediction

```bash
python predict.py
```

* Show ASL gesture in front of webcam
* Prediction appears on screen
* Press **Q** to exit

---

## 📊 Model Architecture

* Input Layer: 63 features
* Hidden Layer 1: Dense (128 neurons, ReLU)
* Hidden Layer 2: Dense (64 neurons, ReLU)
* Output Layer: Softmax
* Optimizer: Adam
* Loss Function: Sparse Categorical Crossentropy

---

## 📈 Results

* Real-time ASL recognition achieved
* Accurate prediction for trained alphabets
* Low latency and smooth execution

---

## ⚠️ Limitations

* Supports limited alphabets (A, B, C)
* Sensitive to lighting and hand orientation
* Background noise may affect detection

---

## 🔮 Future Scope

* Support full ASL alphabet (A–Z)
* Word and sentence formation
* LSTM / CNN-LSTM based temporal modeling
* Text-to-speech conversion
* Web deployment using Streamlit or Flask

---

## 🎓 Project Type

* Academic / Mini Project / Final Year Project
* Domain: Artificial Intelligence & Computer Vision

---

## 👩‍💻 Author

**Mahek**
AI & Data Science Student

---

## 📜 License

This project is intended for educational and academic purposes only.

```
