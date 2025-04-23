# Plant-Growth-Data-Classification-

<div align="center">
  <img src="image.png" alt="Plant Growth Data Classification" width="400"/>
  <h1>🌱 Plant Growth Data Classification 🌾</h1>
  <p>Recommending optimal crop types based on soil conditions using machine learning.</p>
</div>

## 🌿 Overview
This project leverages machine learning to recommend the most suitable crop types based on soil nutrient levels (N, P, K), temperature, humidity, pH, and rainfall. By analyzing these factors, the model helps farmers make informed decisions to optimize crop yield and agricultural productivity.

## 🎯 Problem Statement
Matching crop nutrient requirements with soil conditions is crucial for agricultural productivity. This project addresses the challenge by using a dataset of soil properties and labeled crop types to train a predictive model. The model suggests the best crop for specific soil conditions.

## 📊 Dataset
- **Size:** 2200 entries
- **Features:**
  - N (Nitrogen content)
  - P (Phosphorus content)
  - K (Potassium content)
  - Temperature (°F)
  - Humidity (%)
  - pH (soil acidity/basicity)
  - Rainfall (mm)
- **Target:** Crop type (22 categories: rice, maize, chickpea, lentil, cotton, coffee, etc.)
- **No missing values**

## ✨ Objectives
- Analyze soil condition requirements for different crops.
- Build and train classification models to predict the optimal crop type.
- Evaluate model accuracy and performance.
- Provide insights for crop rotation and soil amendment strategies.

## 🔍 Exploratory Data Analysis (EDA)
- Libraries used: `Sweetviz` and `AutoViz` for detailed data visualization and analysis.
- Distributions, correlations, and feature importance examined.
- Balanced class distribution confirmed (100 samples per crop type).

## 🤖 Machine Learning Approach
- Libraries: `scikit-learn` (Decision Tree, SVM, Random Forest, Bagging, KNN).
- Data pipelines for preprocessing and model training.
- Evaluation metrics: accuracy score, confusion matrix, and classification reports.

## 🛠️ Usage

### ✅ Prerequisites
- Python 3.x
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`, `sweetviz`, `autoviz`

### ⚙️ Installation
```
pip install numpy pandas matplotlib seaborn scikit-learn sweetviz autoviz
```

### 🚀 Running the Project
1.  **Load the dataset:**
    ```
    import pandas as pd
    df = pd.read_csv("Plan_Growth_recommendation.csv")
    ```
2.  **Perform EDA:**
    ```
    import sweetviz as sv
    report = sv.analyze(df)
    report.show_html('sweetviz_report.html')
    ```
3.  **Train a classifier (example with Random Forest):**
    ```
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report

    X = df.drop('label', axis=1)
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))
    ```
4.  **Predict the best crop:** Use the trained model to predict the best crop for new soil data.

## 🌱 Insights and Applications
- Guides farmers on optimal crop selection based on soil conditions.
- Informs soil amendments or crop rotation for improved soil health and yield.
- Supports sustainable agriculture by optimizing resource use.

## 📜 License
This project is open-source and available under the MIT License.

---

For detailed analysis, source code, and dataset, please refer to the project files.


