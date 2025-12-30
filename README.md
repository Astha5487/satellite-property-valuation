# 🏡 Multimodal House Price Prediction Using Tabular Data and Satellite Imagery

## 📌 Project Overview
This project predicts **residential house prices** by combining **structured tabular property data** with **satellite imagery–based visual context**. Traditional real estate valuation models rely heavily on numerical features such as **house size** and **construction quality**. However, these often fail to capture **neighborhood characteristics** like greenery, urban density, and infrastructure quality.  
This project integrates **CNN-extracted satellite image embeddings** with tabular features to enhance both **predictive performance** and **interpretability**.

---

## 🎯 Project Objectives
- Build a **baseline regression model** using tabular property features  
- Extract **high-level visual embeddings** from satellite images using a **pre-trained ResNet50 CNN**  
- Fuse tabular and visual features into a **unified multimodal feature space**  
- Compare **Tabular-only vs. Tabular + Image models** using **RMSE** and **R²**  
- Provide **interpretability and transparency** through **Grad-CAM visual reasoning**

---

## 📂 Dataset Description

### 📑 Tabular Data (CSV)
The structured dataset includes property attributes such as:
- Size: **`sqft_living`**
- Quality indicators: **`grade`**, **`condition`**
- Amenities: **`view`**, **`waterfront`**
- Spatial features: **`latitude`**, **`longitude`**

These features capture **intrinsic house characteristics** and **locational signals**.

---

### 🛰️ Satellite Imagery
Satellite images were retrieved using **latitude and longitude coordinates**. These images capture:
- **Green cover**
- **Road networks**
- **Urban density**
- **Proximity to water bodies**

This provides **environmental context** that is difficult to encode numerically but significantly impacts property value.

---

## 🧠 Modeling and Feature Extraction

### 🖼️ CNN-Based Image Feature Extraction
A **pre-trained ResNet50** was used as a **fixed feature extractor**. By removing the final classification layer, each satellite image was transformed into a **2,048-dimensional embedding** representing **spatial and environmental patterns**.

---

### 🔗 Multimodal Feature Fusion
To prevent visual features from overwhelming the tabular data:
- The **2,048-dimensional CNN embeddings** were compressed into **50 principal components** using **PCA**
- These were **concatenated with tabular features**
- **Strict row-level alignment** was enforced using **property IDs**, ensuring each house was matched with its correct satellite view

---

## 📊 Model Evaluation and Results
Performance was assessed using:
- **Root Mean Squared Error (RMSE)**
- **R² Score**

Evaluation was conducted in **log-price space** to ensure **numerical stability**.

---

## 📈 Performance Summary

### 🔹 Tabular Baseline Model
- **Features Used:** 9–18 Core Features  
- **RMSE (Log):** 0.1651  
- **R² Score:** 0.8668  

### 🔹 Multimodal Model (Unaligned)
- **Features Used:** Tabular + 2,048-D Raw CNN  
- **RMSE (Log):** 0.1980  
- **R² Score:** 0.8120  

### 🔹 Multimodal Model (Aligned + PCA)
- **Features Used:** Tabular + 50-D Aligned CNN  
- **RMSE (Log):** 0.1585  
- **R² Score:** 0.8812  

### 🔹 End-to-End Neural Network
- **Features Used:** Tabular + ResNet Backbone  
- **RMSE (Log):** 0.2450  
- **R² Score:** 0.7200  

---

## 🔍 Explainability: Confirming Curb Appeal
**Grad-CAM (Gradient-weighted Class Activation Mapping)** was applied to the **last convolutional layer of ResNet50** to visualize the model’s focus. This explainability layer confirmed that the model identifies **economically relevant visual signals**, including:

- **Green Canopies:** Validates the importance of greenery in property valuation  
- **Neighborhood Layout:** Distinguishes between dense urban grids and spacious suburban estates  
- **Waterfront Proximity:** Visually confirms the premium associated with nearby water bodies  

---

## ✅ Conclusion
This project demonstrates that while **tabular data remains the strongest predictor** of house prices, **satellite imagery provides a critical supporting layer**. By properly **aligning images** and **reducing feature noise through PCA**, the multimodal approach achieved an **R² of 0.8812**, outperforming the tabular baseline.  
The integration of **Grad-CAM** ensures that predictions are not only accurate, but also **transparent and grounded in visual reality**.

---

## 🛠️ Tech Stack
- **Python:** Pandas, NumPy, Scikit-learn  
- **PyTorch:** ResNet50  
- **XGBoost**  
- **Grad-CAM:** pytorch-grad-cam  
- **Visualization:** Matplotlib, Seaborn
