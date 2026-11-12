# Road Damage and Pothole Detection System 🛣️🤖

An automated infrastructure assessment system built using Machine Learning and Deep Learning to detect road damage and potholes from images, ensuring safe transportation and proactive road maintenance.


📂 Project Folder Structure
Road-Damage-Detection-Project/
│
├── dataset/                    # Dataset directory
│   ├── normal/                 # Normal road images
│   └── pothole/                # Road damage / pothole images
│
├── models/                     # Saved model checkpoints
│   └── pothole_cnn_model.h5    # Trained CNN Deep Learning model
│
├── outputs/                    # Generated outputs & evaluation figures
│   ├── figures/                
│   │   └── class_distribution.png # EDA class distribution chart
│   └── model_comparison.csv    # Performance comparison metrics
│
├── src/                        # Source code modules
│   ├── eda.py                  # Exploratory Data Analysis & visualization
│   ├── preprocessing.py        # Image preprocessing pipeline
│   ├── train_models.py         # Model training script for CNN, SVM, & RF
│   └── evaluate.py             # Model evaluation & metrics script
│
├── app.py                      # Terminal-based prediction script
├── app_ui.py                   # Streamlit web application interface
├── requirements.txt            # Python dependencies and packages
└── README.md                   # Project documentation

Algorithms Implemented
To achieve high accuracy and robust evaluation, the project incorporates three distinct algorithms:
1. Convolutional Neural Network (CNN): A deep learning model optimized for spatial feature extraction and image classification directly from road pixels.
2. Support Vector Machine (SVM): A classical machine learning classifier utilized for baseline feature-based performance evaluation.
3. Random Forest Classifier: An ensemble learning method combining multiple decision trees for robust classification.

✨ Key Features

Interactive Web Interface: User-friendly Streamlit web app (app_ui.py) allowing users to upload road images and get instant damage detection results.
Exploratory Data Analysis (EDA): Automated generation of dataset distribution charts and visual metrics.
Multi-Model Support: Integration of both deep learning (CNN) and traditional ML classifiers (SVM, Random Forest).
Modular Codebase: Cleanly structured source code separated into dedicated modules for training, evaluation, and preprocessing.

⚙️ Installation & Setup
1.Clone the Repository:
git clone <your-repository-url>
cd Road-Damage-Detection-Project
2.Install Required Dependencies:
pip install -r requirements.txt

🚀 How to Run the Project
Run Exploratory Data Analysis (EDA):
cd src
python eda.py

Launch the Streamlit Web Application:
streamlit run app_ui.py

Run Terminal Predictions:
python app.py
Future Scope
Real-time Video Detection: Integrating live dashcam feeds or drone footage for real-time pothole tracking.
Severity Mapping: Estimating the depth and size of potholes to automatically prioritize repair urgency.
Geotagging: Mapping detected road damages onto a GPS-enabled interactive dashboard.

📝 License:
This project is developed as part of academic and research work. Feel free to use and contribute!