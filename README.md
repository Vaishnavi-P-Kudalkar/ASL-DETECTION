🖐️ ASL-DETECTION

A deep learning–based system for real-time American Sign Language (ASL) alphabet detection using MediaPipe for hand tracking and MobileNetV2 for classification.

📘 Overview

This project detects hand gestures corresponding to ASL alphabets in real time using your webcam.
It combines:

MediaPipe Hands → For detecting and tracking the hand region.

MobileNetV2 (CNN) → For classifying the cropped hand image into one of the ASL alphabet classes.

🧠 Algorithm Used

Hybrid pipeline:

MediaPipe Hands — uses a machine learning–based palm detector (SSD) + 21-point landmark regression model to locate the hand.

MobileNetV2 — a lightweight convolutional neural network architecture used for image classification (trained from scratch on the ASL Alphabet dataset).

Model training:
The MobileNetV2 model was trained for 6 epochs on the ASL Alphabet Dataset (A–Z + special signs like Space/Delete/Nothing) using TensorFlow/Keras.

🗂️ Dataset
Dataset: https://www.kaggle.com/datasets/grassknoted/asl-alphabet

Contains ~87,000 labeled images of 29 classes.

Each image shows a hand gesture corresponding to an ASL alphabet.

📁 Folder structure used:
        asl_detector/
        │
        ├── asl_alphabet/
        │   └── asl_alphabet_train/
        │       ├── A/
        │       ├── B/
        │       ├── C/
        │       ├── ...
        │       └── nothing/
        │
        └── newapp.py
        
⚙️ Model Training
Model training is done using MobileNetV2 with data augmentation.
📊 Training Details

    Epochs: 6
    
    Image size: 128×128
    
    Batch size: 32
    
    Optimizer: Adam (lr = 1e-4)
    
    Loss: Categorical Crossentropy
    
    Validation split: 20%

🧾 The trained model is saved as:
asl_mobilenetv2_generator.h5

🚀 How to Run

1. Clone this repository:
    git clone <repo_link>
    cd ASL-DETECTION
2. Download the ASL dataset from Kaggle and store it in a separate folder (same structure as above).
3. Place the trained model file (asl_mobilenetv2_generator.h5) inside your project directory.
4. Run the real-time detection app:
      python newapp.py
5. The webcam window will open — show ASL signs to the camera, and the predicted letter will appear on the screen.
6. Press 'q' to quit.

🧩 Dependencies
Install the required packages:
    pip install tensorflow opencv-python mediapipe numpy matplotlib
