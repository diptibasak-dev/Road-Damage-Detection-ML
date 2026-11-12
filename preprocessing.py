import os
import cv2
import numpy as np

IMG_SIZE = 128  # Image size for processing

def load_and_preprocess_data(data_dir="dataset"):
    data = []
    labels = []
    categories = ['normal', 'pothole'] # Folder names
    
    for category in categories:
        path = os.path.join(data_dir, category)
        class_num = categories.index(category)
        
        if not os.path.exists(path):
            print(f"Warning: Directory {path} not found!")
            continue
            
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = cv2.imread(img_path, cv2.IMREAD_COLOR)
                if img_array is not None:
                    resized_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                    data.append(resized_array)
                    labels.append(class_num)
            except Exception as e:
                pass
                
    return np.array(data) / 255.0, np.array(labels)

if __name__ == "__main__":
    X, y = load_and_preprocess_data()
    print(f"Total loaded images: {len(X)}")
    if len(X) > 0:
        print("Preprocessing test successful! Data shape:", X.shape)
    else:
        print("Please add some images to dataset/normal and dataset/pothole folders.")