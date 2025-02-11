# Waste Classification using CNN with Transfer Learning

A deep learning model that classifies waste images into Organic and Recyclable categories using transfer learning with VGG16 architecture.

## Overview
This project implements a waste classification system using Convolutional Neural Networks (CNN) and transfer learning techniques. The model achieves high accuracy by leveraging the pre-trained VGG16 network and fine-tuning it for waste classification.

## Try the application first before accessing the code
the link of site:https://waste-analyzer-novukyc46t6pdd5qrj.streamlit.app/

## Model Architecture
- **Base Model:** VGG16 (pre-trained on ImageNet)
- **Additional Layers:**
  - Flatten Layer
  - Dense Layer (512 units) with BatchNormalization
  - Dense Layer (256 units) with BatchNormalization
  - Output Layer (2 units) with Softmax activation
- **Training Strategy:** If you want you can use the additional Two-phase training with transfer learning which is commented in the model

## Features
- Transfer Learning using VGG16
- Advanced Data Augmentation
- Learning Rate Scheduling
- Batch Normalization
- Dropout Layers for regularization
- Two-phase training (freeze-unfreeze strategy)

## Dataset
The model uses the Waste Classification dataset from Kaggle containing:
- Two classes: Organic and Recyclable waste
- Training and Testing splits
- RGB images (224x224 pixels)

## Requirements
tensorflow keras numpy pandas opencv-python matplotlib tqdm kagglehub


## Usage
1. Clone the repository
git clone https://github.com/Raviteja5469/CNN-model-for-Waste-Classification/tree/main

2. pip install -r requirements.txt
text
Copy

3. Run the training script:
python train.py


## Model Performance
- **Training Accuracy:** ~98%
- **Validation Accuracy:** ~97%
- **Test Accuracy:** ~98%

## Training Strategy
1. Initial training with frozen VGG16 layers
2. Fine-tuning of last 5 VGG16 layers
3. Learning rate reduction on plateau
4. Comprehensive data augmentation

## Future Improvements
- Implement additional data augmentation techniques
- Experiment with other pre-trained models
- Add real-time prediction capabilities
- Expand to more waste categories

## License
MIT License

## Acknowledgments
- VGG16 pre-trained model
- Waste Classification dataset creators
- TensorFlow and Keras teams
