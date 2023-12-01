# Traffic Accident Prediction and Deployment

## Overview

This project focuses on predicting traffic accidents using historical data and deploying the trained model. Two machine learning algorithms, Linear Regression and Random Forest, were implemented and compared for accuracy.

## Dataset

The dataset, "Monatszahlen Verkehrsunfälle," obtained from the München Open Data Portal, contains detailed information about accidents, including categories, types, years, and months.

## Mission 1: AI Model

### Linear Regression and Random Forest

Implemented machine learning models to predict traffic accidents for the category 'Alkoholunfälle' and the accident type 'insgesamt' in the year 2021 and month 01. Evaluated the performance of both Linear Regression and Random Forest algorithms, with Random Forest showing superior accuracy.

## Visualization

Utilized data visualization techniques to illustrate the historical trends of accidents per category. Specifically, the focus was on the 'Alkoholunfälle' category.


![accients_per_category](https://github.com/agoswami31/dps-ai-task/assets/62375843/65bb9994-12eb-4638-a120-acfcff0f3b07)

![accident_year_trend](https://github.com/agoswami31/dps-ai-task/assets/62375843/f82c7af5-24d7-4dd6-b2ff-537b25d79ebe)

![yearly_accident_peak](https://github.com/agoswami31/dps-ai-task/assets/62375843/66f695ef-aae9-4bae-8c0b-3ab25ec3436e)

## Result Observation after implementing linear regression as well as random forest

- Predicted Value for January 2021 using Linear Regression: 16.624313752837267
- Predicted Value for January 2021 using Random Forest: 28.6
- difference in predicted value is: 11.975686247162734
- Root Mean Square Value Using Linear Regression is: 12.013895610184823
- Root Mean Squared Error using Random Forest is: 9.048873291362701
- difference in rmse is -2.965022318822122


## Mission 2: Publish and Deploy

### GitHub Repository

The complete source code and visualizations are available in the [GitHub repository]([https://github.com/agoswami31/dps-ai-task/blob/main/dps-challenge.ipynb]).

### Deployment

Deployed the trained model on AWS. The model's endpoint accepts JSON input, containing the year and month, and returns accident predictions.

To run inference with the model make a post request to the url:[http://13.211.222.146:8000/predict] 

