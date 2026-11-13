import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
DATASET_DIR = "dataset"
OUTPUT_DIR = r"C:\Users\dipti\OneDrive\Desktop\Road-Damage-ML-Project\outputs\figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)


records = []
for label in sorted(os.listdir(DATASET_DIR)):
    class_path = os.path.join(DATASET_DIR, label)
    if not os.path.isdir(class_path):
        continue
    
    files = os.listdir(class_path)
    split_idx = int(len(files) * 0.8)
    
    for idx, fname in enumerate(files):
        img_path = os.path.join(class_path, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        h, w, _ = img.shape
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        
        records.append({
            "Damage_Class": label,
            "Split": "Train" if idx < split_idx else "Test",
            "Surface_Contrast": round(float(np.std(gray)), 2),
            "Edge_Density": round(float(np.sum(edges > 0) / (h * w) * 100), 2)
        })

df = pd.DataFrame(records)


fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# ১. Class Distribution
sns.countplot(data=df, x="Damage_Class", ax=axes[0, 0], hue="Damage_Class", palette=["#348ABD", "#5D99C6"], legend=False)
axes[0, 0].set_title("1. Road Damage Class Distribution")

# ২. Train vs Test Split
sns.countplot(data=df, x="Damage_Class", hue="Split", palette=["#5D99C6", "#F28E2B"], ax=axes[0, 1])
axes[0, 1].set_title("2. Train vs Test Split per Class")

# ৩. Edge Density Distribution
sns.histplot(data=df, x="Edge_Density", kde=True, color="#5D99C6", ax=axes[1, 0])
axes[1, 0].set_title("3. Edge Density KDE Distribution")

# ৪. Surface Contrast Box Plot
sns.boxplot(data=df, x="Damage_Class", y="Surface_Contrast", hue="Damage_Class", palette=["#348ABD", "#5D99C6"], ax=axes[1, 1], legend=False)
axes[1, 1].set_title("4. Surface Contrast Box Plot")

plt.tight_layout()


target_file = f"{OUTPUT_DIR}/class_distribution.png"
if os.path.exists(target_file):
    os.remove(target_file)

plt.savefig(target_file, dpi=300)
plt.close()

print("SUCCESS: 4 graphs combined inside figures/class_distribution.png!")