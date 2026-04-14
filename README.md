Stock Market Prediction & Analytics Dashboard

A complete data science + machine learning + interactive dashboard project that analyzes stock market data, engineers financial features, and predicts closing prices using multiple regression models.

Project Overview

This project uses a Kaggle stock market dataset to build a full end-to-end machine learning system, including:

Data cleaning & preprocessing
Financial feature engineering
Multiple ML model training & evaluation
Model comparison and selection
Interactive Streamlit dashboard for exploration and prediction
 Key Features
Data Processing
Parses and cleans stock market time-series data
Handles missing values and duplicates
Sorts data chronologically for time-series consistency
Feature Engineering

Created financial indicators such as:

Daily price range
Price change & percentage change
Moving averages (7-day)
Volatility (rolling std)
Lag features (previous day close & volume)
Time features (day, month, quarter, weekday)
🔹 Machine Learning Models

Trained and compared multiple regression models:

Linear Regression
Ridge Regression
Lasso Regression
Random Forest Regressor
Gradient Boosting Regressor

Evaluation metrics:

R² Score
RMSE (Root Mean Squared Error)
MAE (Mean Absolute Error)
🔹 Model Selection
Automatically selects best-performing model based on R² score
Generates predictions and residual analysis
Feature importance analysis for tree-based models
Visualizations

Built using Matplotlib, Seaborn, Plotly:

Stock price trends over time
Candlestick charts
Correlation heatmaps
Residual plots
Actual vs Predicted comparison
Feature importance charts
Model performance comparison
Interactive Dashboard (Streamlit)

A full interactive web app that allows users to:

Pages:
Overview (price trends, stats, distribution)
Exploratory Data Analysis (EDA)
Machine Learning Models comparison
Predictions & error analysis
Features:
Upload custom CSV files
Real-time model training
Interactive Plotly charts
Download prediction results
Tech Stack
Python
Pandas & NumPy
Scikit-learn
Matplotlib & Seaborn
Plotly
Streamlit

Project Structure
project/
│
├── stock_model.py              # ML training pipeline
├── dashboard.py               # Streamlit app
├── random_stock_dataset.csv   # Kaggle dataset
├── feature_importance.png
├── stock_analysis_dashboard.png
├── predictions.csv
└── README.md
Example Workflow
Load dataset (Kaggle stock data)
Clean and preprocess data
Engineer financial features
Train multiple ML models
Compare performance
Deploy interactive dashboard
Results
Best model automatically selected using R² score
Strong performance on time-series regression task
Visual insights into stock price behavior and volatility
Interactive predictions via dashboard
Key Skills Demonstrated
Machine Learning (Regression models)
Feature Engineering (Financial indicators)
Time Series Data Handling
Model Evaluation & Selection
Data Visualization
Full-stack Data Science (Streamlit app)
Important Note

Dataset used is sourced from Kaggle for educational and project purposes.

Future Improvements
LSTM / Deep Learning models for time series
Real-time stock API integration
Portfolio optimization module
Deployment to cloud (Streamlit Cloud / AWS)
Auto trading signal system
