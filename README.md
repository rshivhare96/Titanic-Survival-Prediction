# Titanic Survival Prediction Project

## Introduction

This project aims to predict the survival of passengers aboard the Titanic using machine learning techniques. The dataset used is the famous Titanic dataset, which includes information about passengers, such as their age, class, fare, sex, and other attributes. The goal is to perform a thorough exploration of the dataset, clean the data, and then use machine learning models to predict whether a passenger survived the disaster.

## Step 1: Data Loading and Inspection

The Titanic dataset consists of 891 entries with 12 columns, providing key details about each passenger, such as:
- **PassengerId**: A unique identifier for each passenger.
- **Survived**: A binary column indicating whether a passenger survived (1) or not (0).
- **Pclass**: The class of the passenger (1, 2, or 3).
- **Age**: The age of the passenger.
- **SibSp**: The number of siblings or spouses aboard.
- **Parch**: The number of parents or children aboard.
- **Fare**: The fare the passenger paid for the ticket.
- **Embarked**: The port where the passenger boarded (C = Cherbourg; Q = Queenstown; S = Southampton).
- **Cabin**: The cabin number (many values are missing).
- **Name**: The name of the passenger.
- **Sex**: The gender of the passenger.

### Dataset Overview:
- **Number of Entries:** 891
- **Missing Values:**
  - Age: 177 missing
  - Cabin: 687 missing
  - Embarked: 2 missing

### Summary Statistics:
- **Age:** The average age of passengers is around 29.7 years.
- **Fare:** The highest fare paid is 512.33 GBP, with an average fare of 32.2 GBP.

## Step 2: Data Cleaning

We handled missing values, dropped irrelevant columns, and converted categorical variables into numerical ones. Here's a summary of the cleaning process:
1. **Missing Values**:
   - The missing values in the 'Age' column were filled with the median age.
   - The 'Cabin' column was dropped due to a high percentage of missing values.
   - The missing 'Embarked' values were imputed with the most frequent embarkation port ('S').
2. **Feature Engineering**:
   - We created a new 'FamilySize' feature by adding the 'SibSp' and 'Parch' columns.
   - We extracted titles from the 'Name' column (e.g., Mr, Mrs, Miss).
   - We applied one-hot encoding to convert the 'Sex', 'Embarked', and 'Title' columns into numerical features.

## Step 3: Exploratory Data Analysis (EDA)

The exploratory analysis helped us understand the distributions of key features and their relationship with survival. Key insights include:

### Key Findings from EDA:
- **Age Distribution**: The majority of survivors were around 29 years old, with a higher frequency in the age group of 20-40 years.
- **Fare Distribution**: The most common fare paid was around 10 GBP, with fewer passengers paying higher fares. Passengers who paid over 100 GBP were fewer than 10.
- **Family Size Distribution**: Most passengers traveled alone (single travelers), while larger families (7-8 members) were much less common.
- **Survival Rate by Pclass**: Passengers in Pclass 1 had the highest survival rate (64%), followed by Pclass 2 (47%) and Pclass 3 (24%).
- **Survival Rate by Sex**: Women had a significantly higher survival rate (around 74%) compared to men (around 19%).

### Correlation Analysis:
- FamilySize and SibSp had a high correlation (0.89).
- Sex_male had a negative correlation with Survived (-0.54), suggesting that men were less likely to survive.
- Fare and Pclass had a moderate negative correlation (-0.55), indicating that passengers in higher classes paid higher fares.

## Step 4: Model Building and Evaluation

After data cleaning and EDA, we implemented machine learning models to predict survival. Two models were tested: Logistic Regression and Random Forest.

### Model 1: Logistic Regression
- **Accuracy**: 81.01%
- **Confusion Matrix**:
  - True Negatives: 88
  - False Positives: 17
  - False Negatives: 17
  - True Positives: 57
- **Classification Report**:
  - Precision: 0.77 for survivors, 0.84 for non-survivors.
  - Recall: 0.77 for survivors, 0.84 for non-survivors.

### Model 2: Random Forest
- **Accuracy**: 83.24%
- **Confusion Matrix**:
  - True Negatives: 89
  - False Positives: 16
  - False Negatives: 14
  - True Positives: 60
- **Classification Report**:
  - Precision: 0.79 for survivors, 0.86 for non-survivors.
  - Recall: 0.81 for survivors, 0.85 for non-survivors.

### Cross-validation:
- **Logistic Regression Mean CV Score**: 81.14%
- **Random Forest Mean CV Score**: 79.69%

## Conclusion

From the EDA and modeling, we derived the following key insights:
- **Pclass**: Higher class passengers had higher survival rates.
- **Sex**: Female passengers had a much higher chance of survival than males.
- **Family Size**: Passengers traveling with family had a better survival rate than those traveling alone.

Overall, the **Random Forest** model outperformed **Logistic Regression**, with an accuracy of 83.24%. Both models show good predictive power, but there is room for improvement by fine-tuning parameters or using more advanced models.

This project demonstrates how machine learning can be used to predict survival outcomes based on passenger data, and can be extended further by refining features and exploring other algorithms.
