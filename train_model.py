import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Then your existing code starts here
"""
Complete Image Tagging System using TensorFlow and CIFAR-10 Dataset
...
""""""
Complete Image Tagging System using TensorFlow and CIFAR-10 Dataset
Author: For Internship Project
Date: October 2025
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os

# Create directories for saving models and plots
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("=" * 60)
print("IMAGE TAGGING SYSTEM - CIFAR-10")
print("=" * 60)
print("\n📦 Step 1: Loading CIFAR-10 Dataset...")

# Load CIFAR-10 dataset (built into TensorFlow)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Class names
class_names = ['airplane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f"✅ Dataset loaded successfully!")
print(f"   Training images: {x_train.shape[0]}")
print(f"   Testing images: {x_test.shape[0]}")
print(f"   Image shape: {x_train.shape[1:]} (32x32 RGB)")
print(f"   Number of classes: {len(class_names)}")

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print("\n📊 Step 2: Visualizing Sample Images...")

# Display sample images
plt.figure(figsize=(12, 6))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]], fontsize=8)
    plt.axis('off')
plt.tight_layout()
plt.savefig('results/sample_images.png', dpi=150, bbox_inches='tight')
print("✅ Sample images saved to 'results/sample_images.png'")
plt.close()

print("\n🏗️  Step 3: Building CNN Model Architecture...")

# Build the CNN model
model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.2),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.3),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.4),
    
    # Dense Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✅ Model architecture created!")
model.summary()

print("\n🔄 Step 4: Setting up Data Augmentation...")

# Data augmentation for training
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Apply augmentation to training data
def augment_data(images, labels):
    images = data_augmentation(images)
    return images, labels

# Create augmented dataset
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(1000).batch(64).map(augment_data).prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

print("✅ Data augmentation configured!")

print("\n🎯 Step 5: Training the Model...")
print("   This will take 15-30 minutes depending on your computer.")
print("   (With GPU: ~5-10 minutes)")

# Callbacks for better training
callbacks = [
    keras.callbacks.ModelCheckpoint(
        'models/best_model.h5',
        save_best_only=True,
        monitor='val_accuracy',
        verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    ),
    keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=8,
        restore_best_weights=True,
        verbose=1
    )
]

# Train the model
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=test_dataset,
    callbacks=callbacks,
    verbose=1
)

print("\n✅ Training completed!")

print("\n📈 Step 6: Plotting Training History...")

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy plot
ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Accuracy', fontsize=12)
ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_history.png', dpi=150, bbox_inches='tight')
print("✅ Training plots saved to 'results/training_history.png'")
plt.close()

print("\n🔍 Step 7: Evaluating Model Performance...")

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
print(f"\n📊 Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📊 Test Loss: {test_loss:.4f}")

# Make predictions
y_pred = model.predict(x_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test.flatten()

# Classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

print("\n🎨 Step 8: Creating Confusion Matrix...")

# Confusion matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Number of Predictions'})
plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('results/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Confusion matrix saved to 'results/confusion_matrix.png'")
plt.close()

print("\n🔮 Step 9: Testing Predictions on Sample Images...")

# Show predictions on random test images
plt.figure(figsize=(15, 8))
indices = np.random.choice(len(x_test), 15, replace=False)

for i, idx in enumerate(indices):
    plt.subplot(3, 5, i + 1)
    plt.imshow(x_test[idx])
    
    true_label = class_names[y_true[idx]]
    pred_label = class_names[y_pred_classes[idx]]
    confidence = np.max(y_pred[idx]) * 100
    
    color = 'green' if true_label == pred_label else 'red'
    plt.title(f'True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)', 
              fontsize=8, color=color)
    plt.axis('off')

plt.tight_layout()
plt.savefig('results/sample_predictions.png', dpi=150, bbox_inches='tight')
print("✅ Sample predictions saved to 'results/sample_predictions.png'")
plt.close()

print("\n💾 Step 10: Saving Final Model...")

# Save the complete model
model.save('models/final_image_tagger.h5')
print("✅ Model saved to 'models/final_image_tagger.h5'")

# Save model in TensorFlow SavedModel format (for deployment)
model.save('models/final_image_tagger_savedmodel')
print("✅ Model also saved in SavedModel format for deployment")

print("\n" + "=" * 60)
print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n📁 Generated Files:")
print("   • models/best_model.h5 - Best model during training")
print("   • models/final_image_tagger.h5 - Final trained model")
print("   • models/final_image_tagger_savedmodel/ - Deployment-ready model")
print("   • results/sample_images.png - Sample dataset images")
print("   • results/training_history.png - Training curves")
print("   • results/confusion_matrix.png - Model performance matrix")
print("   • results/sample_predictions.png - Prediction examples")
print("\n📊 Final Results:")
print(f"   • Accuracy: {test_accuracy * 100:.2f}%")
print(f"   • Total Parameters: {model.count_params():,}")
print("\n✨ Your image tagging model is ready for deployment!")
print("=" * 60)