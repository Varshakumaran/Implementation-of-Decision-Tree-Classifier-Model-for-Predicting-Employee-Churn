# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Libraries and Load Dataset.
2.Preprocess the Data.
3.Split the Dataset.
4.Train the Decision Tree Classifier.
5.Make Predictions and Evaluate the Model

## Program:
```
/*
Developed by: VARSHA K 
RegisterNumber:  212223220122
*/
```
```
import pandas as pd
data=pd.read_csv("Employee.csv")
data
```
## Output:
![image](https://github.com/user-attachments/assets/ff6b646b-d8a6-45c2-aa72-a30dadd9f877)

```
data["left"].value_counts()
```
## output:
![image](https://github.com/user-attachments/assets/c1d7df81-d023-4b46-99f4-2268fa66c75e)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

```
## output:
![image](https://github.com/user-attachments/assets/2df1f2ca-5574-4e5a-b682-7300b2e8bdf5)
```
data["salary"]=le.fit_transform(data["salary"])
data
```
## output:
![image](https://github.com/user-attachments/assets/2ae5d757-c569-4efe-a5bd-d47424e2cce4)
```
x=data[["satisfaction_level","last_evaluation","number_project","time_spend_company"]]
x.head()
```
## output:

![image](https://github.com/user-attachments/assets/92aa602d-cde9-4533-b9ae-59ed605f33bc)

```
y=data["left"]
```
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
```
```
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## output:
![image](https://github.com/user-attachments/assets/44247d72-479e-4915-93e3-bcb3dc83b395)

```

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## output:
![image](https://github.com/user-attachments/assets/b30bcf9c-4a50-4c50-b1f1-64e2aa691ce4)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
