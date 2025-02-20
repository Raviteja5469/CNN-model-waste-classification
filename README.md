# 🚀 Waste Classification using CNN with Transfer Learning

## 📌 Overview
This project implements a **CNN-based waste classification model** using **Transfer Learning** with the VGG16 architecture. The model classifies waste into two categories: **Organic** and **Recyclable**, leveraging a two-phase training strategy (freeze-unfreeze).

---

## 🌟 Key Features
- ✅ **Two-phase training strategy**: Freeze VGG16 layers initially, followed by fine-tuning the last few layers.
- 📈 **High Performance**: Achieves up to **98% accuracy** on test data.
- 🎨 **Extensive Data Augmentation**: Various augmentation techniques applied to improve model robustness.
- 🔮 **Future Potential**: Plans to expand functionality, improve predictions, and cover more waste categories.

---

## 📁 Dataset
The **Waste Classification** dataset was sourced from Kaggle. Key details include:
- 🗂 **Classes**: Organic and Recyclable waste.
- 📦 **Structure**: Pre-defined training and testing splits.
- 🖼 **Format**: RGB images with dimensions of **224x224 pixels**.

---

## 🛠️ Requirements
Ensure the following libraries and frameworks are installed:

```bash
tensorflow keras numpy pandas opencv-python matplotlib tqdm kagglehub
```

---

## 🚀 Usage
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Raviteja5469/CNN-model-for-Waste-Classification.git
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Training Script
```bash
python train.py
```

---

## 📊 Model Performance
| Metric | Accuracy |
|--------|---------|
| **Training Accuracy** | ~95% |
| **Validation Accuracy** | ~92% |
| **Test Accuracy** | ~90% |

---

## 🔍 Training Strategy
1. **Initial Training**: Freeze VGG16 layers.
2. **Fine-tuning**: Unfreeze and train the last 5 VGG16 layers.
3. **Learning Rate Adjustment**: Reduce on plateau.
4. **Data Augmentation**: Apply transformations for robustness.

---

## 🔮 Future Improvements
- 🔄 Implement additional data augmentation techniques.
- 📚 Experiment with other pre-trained models.
- 🖥 Add real-time prediction capabilities.
- 🏷 Expand to more waste categories.

---

## 👥 Contributing
Contributions are welcome! Feel free to fork the repository, make changes, and submit a **Pull Request**.

---

## 📞 Contact
| Platform  | Link |
|-----------|------|
| **Author** | Seguri Raviteja |
| **E-mail** | [ravitejaseguri@gmail.com](mailto:ravitejaseguri@gmail.com) |
| **GitHub** | [Raviteja5469](https://github.com/Raviteja5469) |
| **LinkedIn** | [Seguri Raviteja](https://www.linkedin.com/in/ravi-teja-61190a253) |

---

## 🙌 Acknowledgments
- 🏗 **VGG16 Pre-trained Model**
- 📊 **Waste Classification Dataset Creators**
- 🛠 **TensorFlow and Keras Teams**

