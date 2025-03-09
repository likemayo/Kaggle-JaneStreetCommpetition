# Kaggle-JaneStreetCommpetition

## Overview
This repository contains the code and workflow for the **Jane Street Market Prediction** competition hosted on Kaggle. The competition challenges participants to build a model using real-world trading data, offering a glimpse into the daily challenges of successful trading. The dataset highlights the complexities of modeling financial markets, including:

- **Fat-tailed distributions**
- **Non-stationary time series**
- **Sudden shifts in market behavior**

The goal is to develop a robust model that can effectively predict market movements and optimize trading strategies.

---

## Workflow Summary

### 1. **Setup and Data Preparation**
- Copy necessary files from a specified directory.
- Import essential libraries and set up configurations, including feature columns and model paths.
- Define a custom R2 metric function (`r2_val`) for validation purposes.

### 2. **Feature Engineering**
- Specify feature columns for training.
- Create lag features and other relevant features to improve model performance.

### 3. **Model Definition**
- Import multiple machine learning models, including:
  - `LGBMRegressor`
  - `XGBRegressor`
  - `CatBoostRegressor`
  - `VotingRegressor`
- Define a custom neural network model (`NN`) using **PyTorch Lightning**. The model architecture includes:
  - Batch normalization
  - Activation functions
  - Dropout layers
  - Linear layers

### 4. **Model Training**
- Train the neural network model (`NN`) using the training data.
- Use **Adam optimizer** with a learning rate scheduler to reduce the learning rate on a plateau.
- Calculate the mean squared error (MSE) loss, considering sample weights.

### 5. **Model Evaluation**
- Validate the model by calculating the MSE loss and logging it.
- Compute the custom R2 metric at the end of each validation epoch.
- Log and print metrics at the end of each epoch.

### 6. **Prediction and Submission**
- Generate predictions on the test set or unseen data using the trained models.
- Format predictions according to competition requirements.
- Prepare and submit the final submission file.

---

## Approach

### **Ensemble Learning**
- The notebook uses an **ensemble approach** by combining multiple models, including:
  - LightGBM
  - XGBoost
  - CatBoost
  - A custom neural network
- This approach leverages the strengths of different models to improve overall performance.

### **Custom Neural Network**
- A custom neural network model is defined using **PyTorch Lightning**, allowing for flexible and efficient training.
- The model includes:
  - Batch normalization
  - Activation functions
  - Dropout layers for regularization
  - Linear layers

### **Custom Metrics**
- A custom R2 metric function (`r2_val`) is defined for validation, ensuring accurate evaluation of model performance.

### **Feature Engineering**
- The notebook includes feature engineering steps, such as creating **lag features**, which are crucial for time-series data in trading.

### **Optimization and Regularization**
- The neural network uses:
  - **Adam optimizer** with a learning rate scheduler.
  - **Dropout layers** to prevent overfitting.

---
## Result
Our team, Just4silver, achieved a competitive score of 0.008141 in the Jane Street Market Prediction competition. As of the latest leaderboard update, we are ranked 109 out of 3412 teams, earning us a Silver Medal. This achievement reflects the effectiveness of our ensemble modeling approach and feature engineering strategies in tackling the challenges of financial market prediction.

## Repository Structure