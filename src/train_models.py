import tensorflow as tf
from tensorflow.keras import layers, models
import os

# 1. Image Data Generator with Augmentation for better learning
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2 # 20% data validation এর জন্য ভাগ করা হলো
)

train_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    'dataset',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# 2. Building an Advanced CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5), # Model যাতে মুখস্থ না করে, তাই ড্রপআউট
    layers.Dense(1, activation='sigmoid') # Binary output (Normal vs Pothole)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

print("\nStarting Advanced Model Training...")
# 3. Training with more Epochs (15 epochs)
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=val_generator
)

# 4. Save the improved model
model.save("pothole_cnn_model.h5")
print("\nAdvanced Model trained and saved successfully as 'pothole_cnn_model.h5'!")