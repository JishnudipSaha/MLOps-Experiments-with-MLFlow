import pandas as pd
import numpy as np
import mlflow
import logging
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

max_depth = 15
n_estimator = 6


# loading dataset
wine = load_wine()
X = wine.data
y = wine.target

# setting the experiment
mlflow.set_experiment('Experiment-3')

from sklearn.ensemble import RandomForestRegressor
rfc = RandomForestClassifier()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

with mlflow.start_run():
    rf = RandomForestClassifier(max_depth=max_depth)