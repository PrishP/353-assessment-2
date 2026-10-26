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
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

traindf = pd.read_csv("US Airline Data/arrival_delay_data.csv")


x = traindf[["DepDelay"]]
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

xg_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss')
xg_model.fit(pd.DataFrame(x_train), y_train)
y_pred_xg = xg_model.predict(x_test)
print("XGBoost Classifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_xg))
print("Classification Report:\n", classification_report(y_test, y_pred_xg))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xg))

y_prob_rf = rf_model.predict_proba(x_test) # Random Forest
y_prob_lin = lin_model.predict_proba(x_test) # Logistic Regression
y_prob_xg = xg_model.predict_proba(x_test) # XGBoost

#regr = linear_model.LinearRegression()
#regr.fit(x, y)

#calculate the confusion matrix 
cm1 = confusion_matrix(y_test,y_pred_lin) 
#configure the plot 
plt.figure(figsize=(8,6)) 
#create the heat map 
sns.heatmap(cm1,annot=True, fmt='d', cmap='Blues') 
#define labels 
plt.ylabel('Actual') 
plt.xlabel('Predicted')
#add a title 
plt.title('Confusion Matrix Logistic Regression') 
plt.show()

#calculate the confution matrix 
cm2 = confusion_matrix(y_test,y_pred_rf) 
#configure the plot 
plt.figure(figsize=(8,6)) 
#create the heat map 
sns.heatmap(cm2,annot=True, fmt='d', cmap='Blues') 
#define labels 
plt.ylabel('Actual') 
plt.xlabel('Predicted')
#add a title 
plt.title('Confusion Matrix Random Forest') 
plt.show()

#calculate the confution matrix 
cm3 = confusion_matrix(y_test,y_pred_xg) 
#configure the plot 
plt.figure(figsize=(8,6)) 
#create the heat map 
sns.heatmap(cm3,annot=True, fmt='d', cmap='Blues') 
#define labels 
plt.ylabel('Actual') 
plt.xlabel('Predicted')
#add a title 
plt.title('Confusion Matrix XGBoost') 
plt.show()

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lin[:, 1], pos_label=1)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xg[:, 1], pos_label=1)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf[:, 1], pos_label=1)
plt.figure(figsize=(10, 7))
plt.plot(fpr_lr, tpr_lr, linestyle='-', label='Logistic Regression (AUC = {:.3f})'.format(roc_auc_score(y_test, y_prob_lin, multi_class='ovr')))
plt.plot(fpr_rf, tpr_rf, linestyle='-', label='Random Forest (AUC = {:.3f})'.format(roc_auc_score(y_test, y_prob_rf, multi_class='ovr')))
plt.plot(fpr_xgb, tpr_xgb, linestyle='-', label='XGBoost (AUC = {:.3f})'.format(roc_auc_score(y_test, y_prob_xg, multi_class='ovr')))

# Plot the diagonal line representing random guessing 
plt.plot([0, 1], [0, 1], linestyle='--', color='gray') 
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate') 
plt.title('ROC Curves for Random Forest, XGBoost, and Logistic Regression') 
plt.legend() 
# Show the plot 
plt.show()
