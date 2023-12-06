"""
Name:       Min Pan
Email:      Min.Pan99@hunter.cuny.edu
Resources:  Use Kaggle for reffereance
Title:      AirbnbPricePredictor
URL:        https://github.com/Minp45/AirbnbPricePredictor
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier

def load_data(file_name):
    """
    Load the data from the csv file
    """
    data = pd.read_csv('../data/raw/AB_NYC_2019.csv')
    return data
    
def clean_data(data):
    """
    Clean the data
    """
    data.drop(['name','id','host_id', 'host_name','last_review'], axis=1, inplace=True)
    data['reviews_per_month'].fillna(0, inplace=True)
    return data

def filter_data(data):
    data = data.groupby("neighbourhood").filter(lambda x: x['neighbourhood'].count() > 200)
    return data

def encode_data(data):
    data['neighbourhood_group']= data['neighbourhood_group'].astype("category").cat.codes
    data['neighbourhood'] = data['neighbourhood'].astype("category").cat.codes
    data['room_type'] = data['room_type'].astype("category").cat.codes
    return data

def futrue_engineering(data):
    feature_columns=['neighbourhood_group','room_type','price','minimum_nights','number_of_reviews']
    all_data=data[feature_columns]
    all_data['room_type']=all_data['room_type'].factorize()[0]
    all_data['neighbourhood_group']=all_data['neighbourhood_group'].factorize()[0]
    return all_data
    
def split_data(data, X, y):
    x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.1,random_state=105)
    return x_train,x_test,y_train,y_test

def find_best_model(X, y):
    cv = ShuffleSplit(n_splits=4, test_size=0.2, random_state=0)

    print("Finding the best model...")
    print("checking Linear Regression")
    linear_regression_scores = cross_val_score(LinearRegression(), X, y, cv=cv)
    print("checking Ridge Regression")
    ridge_scores = cross_val_score(Ridge(), X, y, cv=cv)
    print("checking Lasso")
    lasso_scores = cross_val_score(Lasso(), X, y, cv=cv)
    print("checking Random Forest")
    random_forest_scores = cross_val_score(RandomForestClassifier(), X, y, cv=cv)

    linear_regression_avg = linear_regression_scores.mean()
    ridge_avg = ridge_scores.mean()
    lasso_avg = lasso_scores.mean()
    random_forest_avg = random_forest_scores.mean()

    print(f"Linear Regression Average Score: {linear_regression_avg}")
    print(f"Ridge Regression Average Score: {ridge_avg}")
    print(f"Lasso Average Score: {lasso_avg}")
    print(f"Random Forest Average Score: {random_forest_avg}")

    best_score = max(linear_regression_avg, ridge_avg, lasso_avg, random_forest_avg)
    best_model = ''
    if best_score == linear_regression_avg:
        best_model = 'Linear Regression'
    elif best_score == ridge_avg:
        best_model = 'Ridge Regression'
    elif best_score == lasso_avg:
        best_model = 'Lasso'
    else:
        best_model = 'Random Forest'

    print(f"\nThe best model is {best_model} with an average score of {best_score}")
    
def fit_model(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=10)
    
    model = LinearRegression()
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))

    return model

def predict_price(model, neighbourhood_group, room_type, minimum_nights, number_of_reviews):
    x = np.zeros(len(model.feature_names_in_))  # Assuming model.feature_names_in_ is available
    x[0] = neighbourhood_group
    x[1] = room_type
    x[2] = minimum_nights
    x[3] = number_of_reviews

    return model.predict([x])[0]

def main():
    data = load_data('../data/raw/AB_NYC_2019.csv')
    print("Initial Data")
    print(data.head())
    print(data.describe())
    print(data.info())
    print("check null")
    data = clean_data(data)
    print(data.isnull().sum())
    print(data['neighbourhood_group'].value_counts())
    
    sns.countplot(x="neighbourhood_group", data=data)
    plt.title("Neighbourhood Group",size=15, weight='bold')
    
    data = filter_data(data)
    data = encode_data(data)
    data = futrue_engineering(data)
    y = data['price']
    X = data.drop(['price'],axis=1)
    
    find_best_model(X, y)
    model = fit_model(X, y)

    print("Prediction #1: ", predict_price(model, 0, 0, 1, 45))
    print("Prediction #2: ", predict_price(model, 0, 1, 3, 0))
    
    
if __name__ == "__main__":
    main()