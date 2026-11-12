import os
import cv2
import numpy as np
import tensorflow as tf

def predict_pothole(image_path):
    model_path = "pothole_cnn_model.h5"
    if not os.path.exists(model_path):
        print("Error: Trained model not found! Please run train_models.py first.")
        return
        
    print("Loading trained CNN model...")
    model = tf.keras.models.load_model(model_path)
    
    if not os.path.exists(image_path):
        print(f"Error: Image path '{image_path}' not found.")
        return
        
    # Read and preprocess the input image
    img = cv2.imread(image_path)
    resized_img = cv2.resize(img, (128, 128)) / 255.0
    input_data = np.expand_dims(resized_img, axis=0)
    
    # Prediction
    prediction = model.predict(input_data)
    score = prediction[0][0]
    
    print("\n--- Prediction Result ---")
    if score > 0.5:
        print(f"Result: Pothole Detected! (Damaged Road) [Confidence: {score:.2f}]")
    else:
        print(f"Result: Normal Road Surface. [Confidence: {1 - score:.2f}]")

if __name__ == "__main__":
    # Test with an image path (you can change the image name/path)
    test_img = "dataset/pothole/road damage1.jpeg"  # Replace with any image path you want to test
    predict_pothole(test_img)