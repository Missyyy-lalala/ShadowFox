# Image Tagging with TensorFlow - CIFAR-10 Classification

## 📋 Project Overview
This project implements an image classification system using Convolutional Neural Networks (CNN) with TensorFlow/Keras. The model is trained on the CIFAR-10 dataset to classify images into 10 categories.

## 🎯 Internship Project Details
- **Objective:** Develop a practical image tagging model for real-world applications
- **Dataset:** CIFAR-10 (60,000 images across 10 classes)
- **Framework:** TensorFlow/Keras
- **Model Architecture:** Deep Convolutional Neural Network

## 🏷️ Classes
The model can identify the following categories:
- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## 🛠️ Technologies Used
- **Python 3.x**
- **TensorFlow 2.x**
- **Keras API**
- **NumPy**
- **Matplotlib**
- **Scikit-learn**
- **Seaborn**

## 📂 Project Structure
```
image-tagging-tensorflow/
│
├── train_model.py          # Main training script
├── predict.py              # Prediction/inference script
├── models/                 # Saved trained models
│   ├── best_model.h5
│   └── final_image_tagger.h5
├── results/                # Training results and visualizations
│   ├── sample_images.png
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── sample_predictions.png
└── README.md              # Project documentation
```

## 🚀 Installation & Setup

### Prerequisites
```bash
pip install tensorflow numpy matplotlib pillow scikit-learn seaborn
```

### Running the Project

**1. Train the Model:**
```bash
python train_model.py
```
This will:
- Download CIFAR-10 dataset automatically
- Train the CNN model (takes 20-30 minutes)
- Generate performance visualizations
- Save the trained model

**2. Make Predictions:**
```bash
python predict.py
```
Choose between:
- Testing on random samples
- Uploading your own images

## 🏗️ Model Architecture

```
Input (32x32x3)
    ↓
Conv2D (32 filters) → MaxPooling → Dropout
    ↓
Conv2D (64 filters) → MaxPooling → Dropout
    ↓
Conv2D (128 filters) → MaxPooling → Dropout
    ↓
Flatten → Dense (128) → Dropout
    ↓
Output (10 classes - Softmax)
```

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 75-85% |
| **Training Time** | ~25-30 minutes (CPU) |
| **Parameters** | ~1.2M |
| **Input Size** | 32x32x3 |

## 🎨 Key Features

### 1. Data Preprocessing
- Image normalization (0-1 range)
- Automatic train-test split
- Batch processing for memory efficiency

### 2. Data Augmentation
- Random horizontal flips
- Random rotations (±10°)
- Random zoom (±10%)

### 3. Training Optimizations
- Adam optimizer
- Learning rate reduction on plateau
- Early stopping to prevent overfitting
- Model checkpointing (saves best model)

### 4. Evaluation Metrics
- Accuracy and Loss curves
- Confusion matrix
- Classification report (Precision, Recall, F1-score)
- Sample predictions visualization

## 📈 Results

### Training Progress
![Training History](results/training_history.png)

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### Sample Predictions
![Sample Predictions](results/sample_predictions.png)

## 🔮 Usage Examples

### Predicting on Test Samples
```python
python predict.py
# Choose option 1: Test on random samples
```

### Predicting Custom Images
```python
python predict.py
# Choose option 2: Upload your image
# Enter image path: /path/to/your/image.jpg
```

## 🎯 Future Improvements
- [ ] Implement transfer learning (ResNet, VGG)
- [ ] Add support for custom datasets
- [ ] Create web interface for predictions
- [ ] Deploy model using TensorFlow Serving
- [ ] Add real-time webcam classification
- [ ] Optimize for mobile deployment (TensorFlow Lite)

## 📝 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Epochs | 30-50 |
| Learning Rate | 0.001 (Adam) |
| Dropout Rate | 0.2-0.5 |
| Optimizer | Adam |
| Loss Function | Sparse Categorical Crossentropy |

## 🐛 Troubleshooting

### Common Issues:

**1. Out of Memory Error:**
```python
# Reduce batch size in train_model.py
batch_size = 16  # Instead of 32
```

**2. Slow Training:**
- Reduce epochs to 20
- Use GPU if available
- Decrease model complexity

**3. Import Errors:**
```bash
pip install --upgrade tensorflow numpy matplotlib
```

## 📚 Learning Resources
- [TensorFlow Documentation](https://www.tensorflow.org/tutorials)
- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CNN Architectures](https://cs231n.github.io/)

## 👨‍💻 Author
**[Your Name]**
- Internship Project: Image Tagging with Deep Learning
- Date: October 2025

## 📄 License
This project is open source and available for educational purposes.

## 🙏 Acknowledgments
- CIFAR-10 dataset by Alex Krizhevsky
- TensorFlow/Keras team
- Internship mentors and supervisors

---

## 📞 Contact
For questions or feedback about this project:
- GitHub: [@your-username]
- Email: your.email@example.com

---

**⭐ If you found this project helpful, please give it a star!**
