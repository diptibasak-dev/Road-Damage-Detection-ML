import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from preprocessing import load_and_preprocess_data

def run_training():
    print("Loading and preprocessing dataset...")
    X, y = load_and_preprocess_data()
    
    if len(X) == 0:
        print("Error: No images found in dataset folders. Please add images first.")
        return

    # Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Flatten images for classical ML models (SVM & Random Forest)
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    # --- MODEL 1: Support Vector Machine (SVM) ---
    print("\n--- Training Model 1: SVM ---")
    svm_clf = SVC(kernel='linear', random_state=42)
    svm_clf.fit(X_train_flat, y_train)
    y_pred_svm = svm_clf.predict(X_test_flat)
    print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
    
    # --- MODEL 2: Random Forest Classifier ---
    print("\n--- Training Model 2: Random Forest ---")
    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train_flat, y_train)
    y_pred_rf = rf_clf.predict(X_test_flat)
    print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
    
    # --- MODEL 3: Deep Learning (CNN) ---
    print("\n--- Training Model 3: Convolutional Neural Network (CNN) ---")
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(1, activation='sigmoid') # Binary classification: Normal vs Pothole
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=16)
    
    # Evaluate CNN
    loss, cnn_acc = model.evaluate(X_test, y_test)
    print(f"\nCNN Test Accuracy: {cnn_acc * 100:.2f}%")
    
    # Save the trained CNN model for prototype demo
    model.save("pothole_cnn_model.h5")
    print("\nTrained CNN model saved successfully as 'pothole_cnn_model.h5'!")

if __name__ == "__main__":
    run_training()