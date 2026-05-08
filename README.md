# Sales_Revenue_Prediction
# Marketing Sales Revenue Prediction using KNN Regression

## Project Overview

This project predicts sales revenue using marketing campaign data and Machine Learning techniques.  
The model is built using the K-Nearest Neighbors (KNN) Regression algorithm and deployed using Streamlit.

It follows a complete end-to-end Machine Learning workflow including data preprocessing, exploratory data analysis, model building, tuning, and deployment.

---

## Objective

To predict sales revenue based on marketing factors such as:

- Ad Spend
- Price
- Discount Rate
- Market Reach
- Impressions
- Click Through Rate
- Competition Index
- Customer Segment
- Marketing Channel
- Product Category

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Pickle

## Dataset

The dataset used in this project is an E-commerce Marketing and Sales dataset containing information related to advertising, pricing, discounts, customer segmentation, and sales performance.

The dataset consists of:
- train.csv
- test.csv

Both datasets are merged and then split using train_test_split for model training and evaluation.

## Workflow

### 1. Data Collection
Marketing dataset containing campaign performance and sales revenue.

### 2. Data Preprocessing
- Converted date column to datetime
- Handled missing values using median and mode
- Standardized inconsistent categorical values
- Removed duplicates
- One-Hot Encoding for categorical variables

### 3. Feature Engineering
Extracted time-based features:
- Year
- Month
- Day
- Weekday
- Hour

### 4. Exploratory Data Analysis (EDA)
Analyzed:
- Feature distributions
- Correlation between variables
- Marketing channel performance
- Pricing and discount impact
- Customer segment behavior

### 5. Model Building
Used K-Nearest Neighbors Regressor (KNN)

- Tested Uniform and Distance weights
- Evaluated K values from 1 to 120
- Selected best parameters based on performance

### 6. Hyperparameter Tuning
- Best K value: 32
- Best weight: Distance

### 7. Model Evaluation
Metrics used:
- MAE (Mean Absolute Error)
- MSE (Mean Squared Error)
- RMSE
- R² Score

### Final Performance:
- MAE ≈ 35.33
- R² Score ≈ 0.23

---

## Observations

- Click Through Rate has the strongest positive impact on revenue
- Higher competition reduces sales
- Influencer and Social Media channels perform best
- Price has a negative impact on sales
- Dataset contains noise and weak correlations

---

## Model Deployment

The model is deployed using Streamlit.

Users can:
- Enter marketing input values
- Get predicted sales revenue instantly

---

## Project Structure

```text
project_folder/
│
├── app.py
├── Code.ipynb
├── knn_model.pkl
├── scaler.pkl
├── columns.pkl
├── requirements.txt
└── README.md
```
## How to Run the Project
```
pip install -r requirements.txt
streamlit run app.py
```

## Future Improvements
- Use Random Forest or XGBoost for better accuracy
- Improve feature engineering
- Handle outliers more effectively
- Apply feature selection techniques
- Deploy on cloud platforms
# Conclusion

## This project demonstrates a complete Machine Learning pipeline:

- Data preprocessing
- Exploratory Data Analysis
- Model training
- Hyperparameter tuning
- Deployment using Streamlit

It provides practical experience in building and deploying regression-based ML applications.
