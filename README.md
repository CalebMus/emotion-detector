# emotion-detector

A real-time facial emotion recognition system built with TensorFlow and OpenCV. The system can detect and classify seven different emotions from both video files and webcam feeds.

## Features

- Real-time emotion detection from webcam feed
- Video file processing with emotion detection
- Detection of 7 emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral
- Shows confidence scores for top 2 detected emotions
- Built using the FER2013 dataset
- Supports multiple face detection in a single frame

## Requirements

```
tensorflow==2.13.0
numpy>=1.24.3
pandas>=2.2.3
matplotlib>=3.9.3
seaborn>=0.13.2
scikit-learn>=1.5.2
opencv-python>=4.10.0
```

## Usage

### Training the Model

The script will:
- Load and preprocess the FER2013 dataset
- Train a CNN model with data augmentation
- Save the best model weights during training
- Generate training history plots

## Model Architecture

The emotion detection model uses a CNN architecture with the following key features:
- Multiple convolutional layers with batch normalization
- MaxPooling layers for dimensionality reduction
- Dropout layers to prevent overfitting
- Dense layers for final classification
- Data augmentation during training

## Dataset

The model is trained on the FER2013 dataset, which contains:
- 26,509 grayscale images of faces
- 7 emotion categories
- 48x48 pixel resolution
- Training, validation, and test sets

## Performance

The current model achieves:
- Training accuracy: ~63%
- Validation accuracy: ~61%
- Real-time performance: ~30 FPS on CPU

## Acknowledgments

- FER2013 dataset providers
- TensorFlow and OpenCV communities
- Code by Sreeram Kondapalli and Caleb Musfeldt
