import scipy
from sklearn import linear_model
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb

traindf = pd.read_csv("US Airline Data/arrival_delay_data.csv")


x = traindf[["DepDelay","Distance"]]
y = traindf["Arrival_Disruption_Severity"]
x.info()
y.info()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("x_train :", len(x_train))
print("y_train :", len(y_train))
print("x_test :", len(x_test))
print("y_test :", len(y_test))

lin_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
lin_model.fit(pd.DataFrame(x_train), y_train)
y_pred_lin = lin_model.predict(x_test)
print("Logistic Regression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lin))
print("Classification Report:\n", classification_report(y_test, y_pred_lin))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lin))

#random forest classifier
rf_model = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_model.fit(pd.DataFrame(x_train), y_train)
y_pred_rf = rf_model.predict(x_test)
print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Classification Report:\n", classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

y_prob_rf = rf_model.predict_proba(x_test)[:, 1] # Random Forest
#y_prob_svm = svm_model.decision_function(x_test) # SVM
#y_prob_lr = lr_model.predict_proba(x_test)[:, 1] # Logistic Regression
#y_prob_xgb = xgb_model.predict_proba(x_test)[:, 1] # # XGBoost

#regr = linear_model.LinearRegression()
#regr.fit(x, y)

