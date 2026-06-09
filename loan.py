import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
# 1. Load the Dataset
data=pd.read_csv('loan_approval_dataset.csv')
# 2. Remove whitespace from column names
data.columns = data.columns.str.strip()
# 3. Exploratory Data Analysis & Preprocessing
print(data.head())
#print(data.isnull().sum())
# Handling Categorical variables
le=LabelEncoder()
for col in data.select_dtypes(include=['str']).columns:
    data[col]=le.fit_transform(data[col])
# 4. Feature Selection
#Assuming 'loan_status' is the target variable
X=data.drop('loan_status',axis=1)
y=data['loan_status']
# 5. Data Splitting and Scaling
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
num_cols=['income_annum','loan_amount','self_employed','cibil_score']
scaler=StandardScaler()
X_train[num_cols]=scaler.fit_transform(X_train[num_cols])
X_test[num_cols]=scaler.transform(X_test[num_cols])
# 6. Model Building
# i. Logistic Regression
lr_model=LogisticRegression(max_iter=200)
lr_model.fit(X_train,y_train)
y_pred_lr=lr_model.predict(X_test)
# ii. Random Forest Classifier
rf_model=RandomForestClassifier(n_estimators=100,max_depth=5,random_state=42)
rf_model.fit(X_train,y_train)
y_pred_rf=rf_model.predict(X_test)
print("\nModel Performance summary:")
print("Logistic Regression Report")
print(classification_report(y_test,y_pred_lr))
print("Random Forest Report")
print(classification_report(y_test,y_pred_rf))
# Create the confusion matrix
cm=confusion_matrix(y_test,y_pred_rf)
#7. Visualization of Results(Actual vs Predicted)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')
plt.show()
