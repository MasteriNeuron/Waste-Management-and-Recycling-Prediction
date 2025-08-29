# ♻️ Waste Management and Recycling Prediction Project
<img width="1906" height="916" alt="image" src="https://github.com/user-attachments/assets/21550bc8-1cbd-4964-b960-264a630bb6e0" />

## 📋 Overview
This project builds a **machine learning model** to predict the **Recycling Rate (%)** for waste management in Indian cities using the *Waste Management and Recycling in India dataset (2019–2023)*.  

The dataset includes features like **Waste Generated (Tons/Day)**, **Population Density**, **City/District**, **Waste Type**, and **Disposal Method**.  

👉 The goal is to optimize waste management and **reduce landfill dependency** through accurate predictions, served via a **Flask web app** with a **user-friendly interface**. 🚀  

---

## 🎯 Objectives
- 📈 **Predict Recycling Rates**: Use regression models to forecast Recycling Rate (%) based on waste management features.  
- 🧠 **Enhance Model Performance**: Achieve high R² scores (target ≥ 0.80) using advanced feature engineering and ensemble methods.  
- 🌐 **Deploy a Web Interface**: Provide a Flask-based interface for users to input data and get predictions.  
- 💾 **Ensure Reproducibility**: Serialize models and preprocessing artifacts for consistent training and prediction.  

---

## 🗂️ Project Structure

````
├── 📂 datasets/
│   ├── 📂 raw/
│   │   └── Waste_Management_and_Recycling_India.csv
│   ├── 📂 processed/
│   │   ├── target_encodings.pkl
│   │   ├── trained\_model.pkl
│   │   ├── scaler.pkl
│   │   ├── selector.pkl
│   │   ├── model_name.pkl
│   │   └── hybrid_model_results.csv
├── 📂 src/
│   ├── 📂 config/
│   │   └── config.yaml
│   ├── 📂 data_processing/
│   │   └── preprocess.py
│   ├── 📂 pipelines/
│   │   ├── training_pipelines.py
│   │   ├── prediction_pipelines.py
│   ├── 📂 utils/
│   │   └── helper.py
│   ├── 📂 logger/
│   │   └── logs.py
├── 📂 static/
│   └── style.css
├── 📂 templates/
│   ├── index.html
│   ├── result.html
│   ├── error.html
├── 📂 logs/
│   └── pipeline.log
├── app.py
├── requirements.txt

````

---

## 🚀 Getting Started

### ✅ Prerequisites
- Python 3.8+ (but go with latest version of python i.e. python=3.13) 🐍  
- Virtual environment (recommended)
- ```conda create -p env python=3.13 -y```
- Dependencies listed in `requirements.txt`  

### ⚙️ Installation
```bash
# Clone the repository
git clone <repository-url>
cd waste-management-prediction

# Set up a virtual environment
python -m venv venv
.\venv\Scripts\activate   # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
````

Ensure the dataset file `Waste_Management_and_Recycling_India.csv` is placed in:

```
datasets/raw/
```

---

## ▶️ Running the Project

### 1️⃣ Train the Model

Run the Flask app and trigger training via the web interface:

```bash
python app.py
```

* Go to [[ecorecycle.com](https://ecorecycle-v1eb.onrender.com/)]
* Click **"Train Model"**

Or run the training pipeline standalone:

```bash
python src/pipelines/training_pipelines.py
```

Check:

* Logs: `logs/pipeline.log`
* Artifacts: `datasets/processed/`

  * `target_encodings.pkl` 🗝️
  * `trained_model.pkl` 🤖
  * `scaler.pkl` 📏
  * `selector.pkl` 🔍
  * `model_name.pkl` 🏷️
  * `hybrid_model_results.csv` 📊

---

### 2️⃣ Make Predictions

* Run the Flask app:

  ```bash
  python app.py
  ```
* Go to [[ecorecycle.com](https://ecorecycle-v1eb.onrender.com/)]
* Enter values for features (e.g., Waste Generated, City/District)
* Submit ➝ **Predicted Recycling Rate (%)** ✅

Check logs in `logs/pipeline.log` for details.

---

## 🔧 Key Components

### 1. Data Preprocessing (`preprocess.py`) 🛠️

* **Feature Engineering** → creates **37 features** from 9 input columns:

  * Landfill coordinates → `(Lat, Long)`
  * Ratios → `Waste_Per_Capita`, `Cost_Efficiency`
  * Target encoding for categorical features
  * Log transforms & polynomial features
* **Data Augmentation** → SMOTE-like + noise, grows dataset **850 ➝ \~3400 rows**
* Saves artifacts: `target_encodings.pkl`, `scaler.pkl`, `selector.pkl`

---

### 2. Training Pipeline (`training_pipelines.py`) 🏋️

* Models:

  * Stacking Ensemble → RF, ET, GBM, LightGBM, XGBoost + Linear Regression meta-model
  * Deep XGBoost
* Feature selection → `SelectKBest`
* Outcome:

  * **Stacking Ensemble**: R² = **0.9725**
  * **Deep XGBoost**: R² ≈ **0.7225**
* Saves: `trained_model.pkl`, `model_name.pkl`

---

### 3. Prediction Pipeline (`prediction_pipelines.py`) 🔮

* Applies **same preprocessing** as training
* Scales, selects top features, then predicts using best model
* Ensures **consistent feature set (37 → selected features)**

---

### 4. Flask Web App (`app.py`, `templates/`, `static/`) 🌐

* TailwindCSS + Bootstrap → dark-themed UI
* Endpoints:

  * `/` ➝ Form input
  * `/train` ➝ Train model
  * `/predict` ➝ Prediction result
* Error handling → `error.html`

---

## 🛑 Challenges & Solutions

* ❌ **Low R² scores initially** (\~ -12%)
* ✅ Fixed with **advanced feature engineering, augmentation, and ensembles**
* Achieved **R² = 0.9725** 🎉

---

## 📊 Outcomes

* 📈 **Model Performance**:

  * Stacking Ensemble R² = **0.9725**
  * Deep XGBoost R² ≈ **0.7225**
* 📂 Dataset: Expanded **850 ➝ 3400 rows**
* 🔍 Features: **37 engineered features**
* 🌟 Web App: Flask UI for predictions
* 📜 Logs: Detailed in `logs/pipeline.log`

---

## 🔮 Future Improvements

* 📈 Confidence intervals for predictions
* 📊 Feature importance visualization in UI
* 🌍 Add external features (weather, policies, etc.)
* 🧠 Neural networks for complex interactions
* 🚀 CI/CD automation with MLflow

---

## 🐛 Debugging Tips

* 📜 Check logs → `logs/pipeline.log`
* 📂 Verify artifacts in `datasets/processed/`
* 🧹 Clear cache:

  ```bash
  del src\*.pyc /s
  ```

---

## 🙏 Acknowledgment

**Special Thanks to Mr. Shubham Chaudhary** for contribution to this project.

---

## 📧 Contact

For questions or contributions:
📩 **[shubham.chaudhary@pw.live](mailto:shubham.chaudhary@pw.live)**

Let’s make waste management smarter! ♻️

```
