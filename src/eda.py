import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create figures directory if not exists
os.makedirs("../outputs/figures", exist_ok=True)

# Count images in dataset folders
normal_path = "../dataset/normal"
pothole_path = "../dataset/pothole"

normal_count = len(os.listdir(normal_path)) if os.path.exists(normal_path) else 0
pothole_count = len(os.listdir(pothole_path)) if os.path.exists(pothole_path) else 0

categories = ['Normal Road', 'Pothole']
counts = [normal_count, pothole_count]

# Plotting bar chart
plt.figure(figsize=(6, 4))
sns.barplot(x=categories, y=counts, palette='viridis')
plt.title("Road Image Class Distribution")
plt.ylabel("Number of Images")
plt.savefig("../outputs/figures/class_distribution.png")
plt.close()

print("EDA Bar Chart generated successfully in outputs/figures/!")