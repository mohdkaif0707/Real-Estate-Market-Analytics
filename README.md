# 🏠 Real Estate Market Analytics

An interactive **Real Estate Business Analytics Dashboard** built using Python, Pandas, Plotly, Scikit-learn and Streamlit.

The application analyzes residential property data across multiple Indian cities and provides insights into property prices, BHK trends, property types, locations, area-price relationships and price-per-square-foot trends.

It also includes Machine Learning models for residential property price prediction.

---

## 🌐 Live Demo

🚀 **Live Dashboard:**  
https://real-estate-market-analytics.streamlit.app/

📂 **GitHub Repository:**  
https://github.com/mohdkaif0707/Real-Estate-Market-Analytics

---

## 📊 Project Overview

Real Estate Market Analytics is designed as a business analytics solution that helps users explore residential property data and understand market patterns through interactive visualizations.

The dashboard allows users to:

- Analyze property prices
- Compare cities
- Explore BHK trends
- Analyze property types
- Study area vs price relationships
- Compare price per square foot
- Examine locality characteristics
- Upload custom CSV datasets
- Evaluate Machine Learning models
- Predict estimated property prices

---

## ✨ Key Features

### 📁 Dataset Management

- Default real estate dataset
- Upload custom CSV files
- Automatic column detection and mapping
- Price unit conversion
- Data cleaning
- Duplicate detection
- Missing-value analysis

### 📈 Market Analytics

- Total number of properties
- Average property price
- Median property price
- Average property area
- Average price per square foot
- Property type distribution
- City-wise average price
- BHK distribution
- BHK-wise average price
- Locality type analysis
- Area vs price relationship

### 🏙️ City & Locality Comparison

Users can compare multiple locations using:

- Average Price
- Median Price
- Average Price per Sq.Ft
- Property count
- Price distribution

Interactive Plotly visualizations are used to make the comparison easier.

---

## 🤖 Machine Learning

The project includes two Machine Learning models for property price estimation:

### 1. Linear Regression

A baseline regression model used to understand the relationship between property features and price.

### 2. Random Forest Regressor

An ensemble-based regression model used to capture nonlinear relationships between property characteristics and price.

### Model Evaluation

The models are evaluated using:

- **MAE — Mean Absolute Error**
- **RMSE — Root Mean Squared Error**
- **R² Score — Coefficient of Determination**

The dashboard provides:

- Model comparison
- R² comparison
- MAE comparison
- Actual vs Predicted price visualization
- Feature importance analysis

---

## 🧠 Property Price Prediction

Users can enter property characteristics such as:

- City
- Property Type
- BHK
- Bathrooms
- Super Area
- Carpet Area
- Floor Number
- Total Floors
- Property Age
- Furnishing Status
- Parking
- Lift Availability
- Gated Community
- Distance to Metro
- Distance to City Center

The trained Machine Learning model then generates an estimated property price.

> ⚠️ The prediction is a Machine Learning model estimate and should not be considered an official property valuation.

---

## 🗂️ Dataset

The project uses a residential real estate dataset containing **80,000 property records** across **10 Indian cities**.

### Cities

- Chandigarh
- Kolkata
- Delhi
- Ahmedabad
- Pune
- Hyderabad
- Mumbai
- Bangalore
- Jaipur
- Chennai

### Dataset Features

| Feature | Description |
|---|---|
| Property_ID | Unique property identifier |
| City | Property city |
| Locality_Type | Type of locality |
| Property_Type | Type of property |
| BHK | Number of bedrooms |
| Bathrooms | Number of bathrooms |
| Super_Area_SqFt | Super built-up area |
| Carpet_Area_SqFt | Carpet area |
| Floor_Number | Property floor |
| Total_Floors | Total building floors |
| Age_of_Property | Property age |
| Furnishing_Status | Furnishing condition |
| Parking | Parking availability |
| Lift_Available | Lift availability |
| Gated_Community | Gated community availability |
| Distance_to_Metro_km | Distance from metro |
| Distance_to_City_Center_km | Distance from city center |
| Price_INR_Lakhs | Property price in INR Lakhs |

---

## 🛠️ Technology Stack

### Programming Language

- Python

### Data Analysis

- Pandas
- NumPy

### Data Visualization

- Plotly

### Machine Learning

- Scikit-learn

### Dashboard

- Streamlit

### Version Control

- Git
- GitHub

### Deployment

- Streamlit Community Cloud

---

## 📁 Project Structure

```text
Real-Estate-Market-Analytics/
│
├── data/
│   └── real_estate.csv
│
├── static/
│   └── css/
│       └── style.css
│
├── templates/
│   └── index.html
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md