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

## 📂 Dataset Samples
Below are representative samples from the dataset, showing the visual differences between the three classes:

![Dataset Samples](assets/dataset_samples.png)
*(Figure 1: Representative chest X-ray images showing Normal, Bacterial Pneumonia, and COVID-19 cases)*

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

## 📊 Results

The **Transfer Learning (ResNet18)** approach demonstrated superior performance compared to the custom CNN, achieving higher accuracy and better separation between classes.

### Visual Analysis (t-SNE)
The plot below shows the projection of the model's feature embeddings into 2D space. The clear clustering indicates that the model has successfully learned to distinguish between the three conditions.

![t-SNE Visualization](assets/tsne_plot.png)
*(Figure 2: t-SNE projection of the model's feature embeddings)*

## 🚀 How to Run

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/JacobMaimon13/COVID19-XRay-Classification.git](https://github.com/JacobMaimon13/COVID19-XRay-Classification.git)
    cd COVID19-XRay-Classification
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

5.  **Run Training:**
    ```bash
    python main.py
    ```

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments
* Instructors: Alon Finestein, Thomas Mendelson.
* Data provided by [ieee8023](https://github.com/ieee8023) and [Kaggle](https://www.kaggle.com).
