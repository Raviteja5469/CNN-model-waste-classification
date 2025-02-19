# Waste Classification using CNN with Transfer Learning

## Overview
This project implements a **CNN-based waste classification model** using **Transfer Learning** with the VGG16 architecture. The model classifies waste into two categories: **Organic** and **Recyclable**, leveraging a two-phase training strategy (freeze-unfreeze).

---

## 🌟 Key Features
- **Two-phase training strategy**: Freeze VGG16 layers initially, followed by fine-tuning the last few layers.
- **High Performance**: Achieves up to **98% accuracy** on test data.
- **Extensive Data Augmentation**: Comprehensive techniques are applied to improve robustness.
- **Future Potential**: Plans to expand functionality, improve predictions, and cover more waste categories.

---

## 📁 Dataset
The **Waste Classification** dataset was sourced from Kaggle. Key details include:
- **Classes**: Organic and Recyclable waste.
- **Structure**: Pre-defined training and testing splits.
- **Format**: RGB images with dimensions of **224x224 pixels**.

---

## 🛠️ Requirements
Ensure the following libraries and frameworks are installed:

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

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📞 Contact

| Platform | Link |
|----------|------|
| **Author** | Raviteja |
| **E-mail** | [Raviteja] (ravitejaseguri@gmail.com) |
| **GitHub** | [Raviteja5469](https://github.com/Raviteja5469) |
| **LinkedIn** | [Seguri Raviteja](https://www.linkedin.com/in/ravi-teja-61190a253) |

## 📄 License

This project is open-source and available under the MIT License........

---
## Acknowledgments
- VGG16 pre-trained model
- Waste Classification dataset creators
- TensorFlow and Keras teams
