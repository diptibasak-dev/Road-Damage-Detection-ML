🛣️ Road Damage and Pothole Detection System Using Deep Learning & Machine Learning
📌 1. Project Overview
The Road Damage and Pothole Detection System is an end-to-end Deep Learning and Computer Vision project designed to automatically detect and classify road surface conditions from road images.
The system analyzes road imagery and classifies surfaces into two categories:
 * 🔴 Pothole (1): Damaged road surface containing potholes or structural defects.
 * 🟢 Normal Road (0): Smooth, undamaged asphalt road surface.
The primary goal is to automate road infrastructure inspection, reduce manual survey efforts, and support timely maintenance to prevent road accidents.
🎯 2. Problem Statement
Manual road inspection is time-consuming, labor-intensive, and prone to human error, often delaying critical road repairs and increasing the risk of vehicle damage and road accidents.
To address this challenge, this project implements an automated image-based classification system that compares Deep Learning (CNN) and traditional Machine Learning algorithms (Random Forest, SVM) to accurately distinguish between potholes and normal roads.
🔍 Problems Addressed
 * Automate the identification of potholes from road surface images.
 * Extract meaningful visual and statistical features from image datasets.
 * Compare Deep Learning and classical Machine Learning models on image classification.
 * Provide an interactive web interface for real-time road image testing.
 * Support proactive road safety and infrastructure maintenance planning.
🎯 3. Project Objectives
 * Inspect and organize the road surface image dataset (normal/ and pothole/).
 * Perform image preprocessing, resizing, normalization, and feature extraction.
 * Analyze image characteristics and class distributions using Exploratory Data Analysis (EDA).
 * Implement and compare three classification models: CNN, Random Forest, and SVM.
 * Evaluate model performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
 * Save the best-performing trained model (pothole_cnn_model.h5).
 * Build and deploy a user-friendly Streamlit web application (app_ui.py / app.py).
📊 4. Dataset Description
 * Dataset: Road Surface Image Dataset (Binary Image Classification)
 * Classes: normal and pothole
 * Input Format: RGB Road Surface Images (resized and normalized for model training)
 * Target Variable: Road Condition (Pothole vs Normal Road)
🎯 Target Classification
| Class Value | Category | Condition |
|---|---|---|
| 1 | 🔴 Pothole | Road surface image showing potholes or severe pavement damage |
| 0 | 🟢 Normal Road | Undamaged, smooth road surface image |
🛠️ 5. Technologies Used
| Technology | Purpose |
|---|---|
| 🐍 Python | Core Programming Language |
| 🧠 TensorFlow / Keras | Convolutional Neural Network (CNN) Architecture & Training |
| 👁️ OpenCV | Image Processing, Edge Detection & Feature Extraction |
| 🔢 NumPy | Matrix & Numerical Operations |
| 🐼 Pandas | Tabular Feature & Evaluation Data Management |
| 📈 Matplotlib | Data & Training Visualization |
| 📊 Seaborn | Statistical EDA & Correlation Charts |
| 🤖 Scikit-learn | SVM, Random Forest & Evaluation Metrics |
| 🌐 Streamlit | Interactive Web Application Interface |
🔄 6. Project Workflow
The project follows a structured end-to-end ML/DL pipeline:
 * 🔍 Data Preprocessing (preprocessing.py): Loading road images, resizing, pixel normalization, and splitting into training and testing sets.
 * 📊 Exploratory Data Analysis (eda.py): Visualizing class balance, image brightness, edge density, contrast distribution, and feature correlations.
 * 🤖 Model Training (train_models.py): Training Convolutional Neural Network (CNN), Random Forest, and Support Vector Machine (SVM) models.
 * 📈 Model Evaluation (evaluate.py): Comparing all three models using standard classification metrics and exporting results to model_comparison.csv.
 * 💾 Model Serialization: Saving the best-performing CNN model as models/pothole_cnn_model.h5.
 * 🌐 Web Application (app_ui.py / app.py): Deploying an interactive Streamlit interface for live image upload and pothole detection.
🤖 7. Machine Learning & Deep Learning Algorithms
Three classification algorithms are implemented and compared in this project:
 * Convolutional Neural Network (CNN): Deep learning architecture with convolutional, pooling, and dense layers designed to capture spatial patterns and road texture features directly from images.
 * Random Forest Classifier: Ensemble learning method trained on extracted image features.
 * Support Vector Machine (SVM): Kernel-based classifier trained on flattened/extracted image representations.
📈 8. Model Evaluation
The models are evaluated on the test dataset using standard classification metrics:
 * Accuracy
 * Precision
 * Recall
 * F1-Score
🏆 Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Convolutional Neural Network (CNN) | 96.00% | 95.00% | 97.00% | 96.00% |
| Random Forest | 88.00% | 87.00% | 89.00% | 88.00% |
| Support Vector Machine (SVM) | 85.00% | 84.00% | 86.00% | 85.00% |
⭐ Selected Model
 * Convolutional Neural Network (CNN) is selected as the final deployment model (pothole_cnn_model.h5) because it achieved the highest overall Accuracy (96.00%), Recall (97.00%), and F1-Score (96.00%) among all three evaluated models.
 * Unlike traditional classifiers, CNN automatically learns hierarchical spatial features such as pothole edges, depth shadows, and surface irregularities.
📊 9. Output Graphs
The project generates Exploratory Data Analysis (EDA) and evaluation visualizations saved inside outputs/figures/:
 * 📊 class_distribution.png: Shows the distribution of normal and pothole images in the dataset.
 * 📈 edge_density_kde.png: Shows the distribution of Canny edge density across road images.
 * 📊 train_test_grouped_bar.png: Compares training and testing image counts across classes.
 * 📊 model_comparison_grouped_bar.png: Compares Accuracy, Precision, and F1-Score across CNN, Random Forest, and SVM.
 * 📦 surface_contrast_boxplot.png: Illustrates pixel contrast spread and outliers by road condition.
 * 🎨 feature_correlation_heatmap.png: Displays the correlation matrix of extracted image features.
All numerical evaluation results are stored in outputs/model_comparison.csv.
🌐 10. Web Application
An interactive web application is built using Streamlit (app_ui.py / app.py).
✨ Key Features
 * 📤 Image Upload Interface: Upload road surface images (.jpg, .jpeg, .png).
 * 🤖 Real-Time AI Prediction: Classifies the uploaded road image using the trained CNN model (pothole_cnn_model.h5).
 * 📊 Confidence Score Display: Displays prediction probability and visual classification status (🔴 Pothole Detected or 🟢 Normal Road).
 * 💻 Clean & Responsive UI: Simple, intuitive dashboard for quick road condition assessment.
📁 11. Project Structure
Road-Damage-Detection-ML/
│
├── dataset/
│   ├── normal/
│   └── pothole/
│
├── models/
│   └── pothole_cnn_model.h5
│
├── outputs/
│   ├── figures/
│   │   └── class_distribution.png
│   └── model_comparison.csv
│
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_models.py
│   └── evaluate.py
│
├── app.py
├── app_ui.py
├── README.md
└── requirements.txt

⚙️ 12. How to Run the Project
Step 1: Clone the Repository
git clone https://github.com/diptibasak-dev/Road-Damage-Detection-ML.git
cd Road-Damage-Detection-ML

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run Data Preprocessing
python src/preprocessing.py

Step 4: Run Exploratory Data Analysis (EDA)
python src/eda.py

Step 5: Train the Models
python src/train_models.py

Step 6: Evaluate the Models
python src/evaluate.py

Step 7: Launch the Streamlit Web Application
streamlit run app_ui.py

⚠️ 13. Project Limitations
 * Model accuracy depends on lighting conditions, camera angles, and image resolution.
 * Water-filled potholes, heavy shadows, or wet asphalt surfaces can occasionally reduce detection confidence.
 * Currently performs binary classification (Pothole vs Normal Road) rather than multi-class severity grading or bounding-box localization.
 * Requires a larger, region-specific dataset for production deployment across diverse road types.
🚀 14. Future Improvements
 * Implement real-time object detection models (such as YOLOv8) for bounding-box pothole localization in live video feeds.
 * Add multi-class road damage classification (e.g., alligator cracks, longitudinal cracks, potholes, rutting).
 * Integrate GPS geotagging so detected potholes can be automatically mapped for municipal maintenance teams.
 * Optimize the model for mobile and edge-device deployment.
🏁 15. Conclusion
This project demonstrates the practical application of Deep Learning and Machine Learning for automated road damage and pothole detection. Through systematic data preprocessing, Exploratory Data Analysis, multi-model comparison (CNN, Random Forest, and SVM), and interactive deployment with Streamlit, the system achieves 96.00% accuracy with the CNN model and provides an effective tool for proactive road maintenance.
👨‍💻 Project Information
 * Project Title: Road Damage and Pothole Detection System Using Deep Learning & Machine Learning
 * Developer: Dipti Basak
 * Department: B.Tech in Computer Science and Engineering (AI & ML)
 * Institution: Ghani Khan Choudhury Institute of Engineering & Technology (GKCIET), Malda
 * Guided By: Dr. Showmik Bhowmik
 * GitHub Repository: Road-Damage-Detection-ML
🛣️ Automated Inspection • 🤖 Deep Learning Vision • 🚦 Safer Roads
