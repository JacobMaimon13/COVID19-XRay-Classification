# COVID-19 Chest X-Ray Classification 🩻

![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research-green?style=for-the-badge)

**Authors:** Jacob Maimon, Bar Naor  
**Course:** Deep Learning and Its Applications to Signal and Image Processing (BGU)

---

## 📌 Overview
This project tackles the critical challenge of classifying chest X-ray images into three categories: **Normal**, **Bacterial Pneumonia**, and **COVID-19**. 

A major focus of this research was dealing with **imbalanced data**, as COVID-19 samples were significantly scarcer than other classes. We experimented with various techniques to improve model generalization and utilized **Transfer Learning** to achieve robust results.

## 🔬 Methodology

### 1. Data Preparation
We combined two datasets:
* **COVID-19 Chest X-Ray Dataset** (ieee8023)
* **Chest X-Ray Images (Pneumonia)** (Kaggle/Tolga Dincer)

### 2. Architectures
We implemented and compared two main approaches:
* **Custom SimpleCNN:** A lightweight Convolutional Neural Network built from scratch.
* **Transfer Learning (ResNet18):** Utilizing a pre-trained ResNet18 backbone, fine-tuned for our 3-class problem.

### 3. Handling Imbalance
To address the dataset imbalance, we conducted extensive experiments:
* **Baseline:** Training on raw, imbalanced data.
* **Under-sampling:** Reducing the majority classes.
* **Over-sampling (Augmentation):** Artificially generating COVID-19 samples using geometric transformations.
* **Class Weights:** Adjusting the loss function (CrossEntropy) to penalize errors on the minority class more heavily.

### 4. Explainability & Analysis
* **t-SNE Visualization:** We projected the high-dimensional feature maps into 2D to visualize class separability.
* **Confusion Matrices:** Detailed analysis of misclassifications.

## 📊 Results

The **Transfer Learning (ResNet18)** approach demonstrated superior performance compared to the custom CNN.

### Visual Analysis
Below is the **t-SNE visualization** showing how the model learned to separate the three classes in the latent space. As seen, the classes are well-clustered, indicating robust feature learning.

![t-SNE Visualization](figuers/tsne_plot.png)
*(Figure 1: t-SNE projection of the model's feature embeddings)*

### Classification Performance
The confusion matrix highlights the model's high accuracy, with minimal confusion between COVID-19 and other pneumonia types.

![Confusion Matrix](figuers/confusion_matrix.png)
*(Figure 2: Confusion Matrix on the Test Set)*

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/JacobMaimon13/COVID19-XRay-Classification.git
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download Data:**
    * The code expects data in a `data/` folder.
    * You will need to download the [COVID-19 dataset](https://github.com/ieee8023/covid-chestxray-dataset) and the [Kaggle Pneumonia dataset](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images).

4.  **Prepare the Dataset:**
    Organize the raw data into training and testing CSV files by running:
    ```bash
    python -m src.prepare_data
    ```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
* Instructors: Alon Finestein, Thomas Mendelson.
* Data provided by [ieee8023](https://github.com/ieee8023) and [Kaggle](https://www.kaggle.com).
