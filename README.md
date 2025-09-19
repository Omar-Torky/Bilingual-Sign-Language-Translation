# 🖐️ Sign Language Recognition with Mediapipe & RandomForest

![Arabic Alphabet](Arabic Letters)


This project implements a **real-time Sign Language Recognition system** using [Mediapipe Hands](https://google.github.io/mediapipe/solutions/hands.html) for landmark extraction and a **Random Forest Classifier** for classification.  
It enables users to form words and sentences by signing letters (A–Z) | (أ-ي) along with **Space** and **Backspace** gestures in real-time using a webcam.

---

# 📌 Features
- Capture and preprocess your own dataset of hand gesture images.
- Extract **21 hand landmarks** per hand using Mediapipe.
- Train a **Random Forest Classifier** on extracted landmarks.
- Perform **real-time prediction** using webcam input.
- Dynamically display recognized sentence with a stabilizing sliding-window approach.
- Keyboard controls for enhanced usability:
  - **Q** → Quit  
  - **C** → Clear sentence  
  - **Space** → Switch Language  
  - **Z** → Backspace  

---

# 📂 Project Structure

```bash

├── data/             # Collected images (organized by class)
├── data.pickle       # Preprocessed dataset (features + labels)
├── model.p           # Trained Random Forest model
├── collect_imgs.py   # Collects gesture data via webcam
├── create_dataset.py     # Extracts hand landmarks & saves dataset
├── train_classifier.py    # Trains Random Forest classifier
├── Inference.ipynb      # Real-time gesture recognition and sentence builder
└── README.md
```

---

# ⚙️ Installation
## 1. Clone this repository:
```bash
git clone https://github.com/Omar-Torky/Bilingual-Sign-Language-Translator.git
cd Bilingual-Sign-Language-Translator
```

## 2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows
```

## 3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

# 🚀 Usage
## 1. Collect Data
- Press Q to start capturing for each class.
- Images will be saved under ./data/<class_id>/.
```bash
python collect_imgs.py
```


## 2. Preprocess Data
- Convert collected images into hand landmark datasets
- This generates data.pickle.
```bash
python preprocess.py
```

## 3. Train the Model
- Trains the Random Forest classifier
- Saves trained model as model.p
- Prints training accuracy.
```bash
python train_classifier.py
```


## 4. Run Real-time Recognition
- Start the live webcam recognition:
```bash
python Inference.ipynb
```
---

# 🎮 Controls
- Q → Quit Program
- C → Clear Sentence
- Space → Switch Language
- Z → Backspace

---

# ✨ Example Workflow
1. Collect images for each Letter.
2. Preprocess to generate landmark dataset (data.pickle).
3. Train the Random Forest model (model.p).
4. Use a Webcam for real-time recognition and sentence building.






