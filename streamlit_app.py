import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# read data
df=pd.read_csv("C:\Users\DELL\Desktop\Student-Performance-Prediction\data\student_performance.csv")
# show data
print(df)

# check data......
print("first 5 rows:")
print(df.head())
print("\n")

# (it give row,column)
print("shape of data:")
print(df.shape)
print("\n")

print("name of column:")
print(df.columns)
print("\n")

# infomation data(it give index range,no of column,(column name,no of non null,data type),memory use,None mean this method returnrd no value)
print("information of data:")
df.info()
print("\n")

# show count of non null vale,mean,std,min,25%,50%,75%,max
print("data describe:")
print(df.describe())
print("\n")

print("cheak missing value...")
# show column+no of null value
print("number of null value in row:")
print(df.isnull().sum())
print("\n")
# it show missing value is exist or not
print("missing exist?")
print(df.isnull().values.any())
print("\n")
# ------------------------------
print("fill missing value.....")
# fill missing value with mean for numerical
df["Study_Hours"]=df["Study_Hours"].fillna(df["Study_Hours"].mean())
df["Attendance(%)"]=df["Attendance(%)"].fillna(df["Attendance(%)"].mean())
# fill missing value with mean for orical use[0] because mode give list not one value
df["Grade"]=df["Grade"].fillna(df["Grade"].mode()[0])
print("missing value after cleaning")
print(df.isnull().sum())
print("\n")

print(" check duplicated row.....")
# total no of duplicate row
print("number of duplicated value:")
print(df.duplicated().sum())
print("\n")
# to show duplicated row name
print("name of row has duplicated value:")
print(df[df.duplicated()])
print("\n")
# -----------------------------
print("remove duplicated value....")
df=df.drop_duplicates()
print("\n")
# to check remove or not
print("number duplicated value:")
print(df.duplicated().sum())
print("\n")

print("check incorrect data types..")
print(df.dtypes)
print("\n")

print("check unique value...")
print(df["Student_ID"].unique())

print("label encoding....")
le= LabelEncoder()
df["Gender"]=le.fit_transform(df["Gender"])
print("gender")
print(df["Gender"])
print("\n")
# like female->0 male->1
print("grade")
df["Grade"]=le.fit_transform(df["Grade"])
print(df["Grade"])
print("\n")

plt.figure(figsize=(8,5))
plt.scatter(df["Study_Hours"],df["Test_Score"])
plt.xlabel("Study Hours")
plt.ylabel("Test Score")
plt.title("Study Hours vs Test Score")
plt.show()

df["Grade"].value_counts().plot(kind="bar")
plt.xlabel("Grade")
plt.ylabel("Number of Students")
plt.title("Grade Distribution")
plt.show()


plt.scatter(df["Attendance(%)"],df["Test_Score"])
plt.xlabel("Attendance(%)")
plt.ylabel("Test score")
plt.title("Attendance vs Test score")
plt.show()

from sklearn.model_selection import train_test_split
# fature and target
X=df[["Age","Study_Hours","Attendance(%)"]]
y=df["Test_Score"]
# split data
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
print("X_train:",X_train.shape)
print("X_test:",X_test.shape)
print("y_train:",y_train.shape)
print("y_test:",y_test.shape)
print("\n")

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

result=[]

linear_model=LinearRegression()
linear_model.fit(X_train,y_train)
# predict
y_pred=linear_model.predict(X_test)

result.append(["LINEAR REGRASSION",
               r2_score(y_test,y_pred),
               mean_absolute_error(y_test,y_pred),
               mean_squared_error(y_test,y_pred)])

tree_model=DecisionTreeRegressor(random_state=42)
tree_model.fit(X_train,y_train)
y_pred=tree_model.predict(X_test)

result.append(["decision tree",
               r2_score(y_test,y_pred),
               mean_absolute_error(y_test,y_pred),
               mean_squared_error(y_test,y_pred)])

forest_model=RandomForestRegressor(n_estimators=100,random_state=42)
forest_model.fit(X_train,y_train)
y_pred=forest_model.predict(X_test)

result.append(["random forest",
               r2_score(y_test,y_pred),
               mean_absolute_error(y_test,y_pred),
               mean_squared_error(y_test,y_pred)])

comparison=pd.DataFrame(result,columns=["model","r2","mae","mse"])
print("comparison table")
print(comparison.round(3))

# user input data
def predict_marks():
    age=int(input("enter age:"))
    hours=int(input("enter study hours:"))
    percetage=int(input("enter percentage:"))

    new_data=pd.DataFrame({"Age":[age],
                        "Study_Hours":[hours],
                        "Attendance(%)":[percetage]})

    new_data_scaled=scaler.transform(new_data)

    predicion=forest_model.predict(new_data_scaled)

    print("predicison test score:",predicion[0])
predict_marks()

# Evaluate using:Accuracy,Precision,Recall,F1-score,Confusion Matrix
# for evalute matrix classification need categorical data
X=df[["Age","Study_Hours","Attendance(%)","Test_Score"]]
y=df["Grade"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

print("\naccuracy_score:",accuracy_score(y_test,y_pred))
print("\nprecision_score:",precision_score(y_test,y_pred,average="weighted"))
print("\nrecall_score:",recall_score(y_test,y_pred,average="weighted"))
print("\nf1_score:",f1_score(y_test,y_pred,average="weighted"))
print("\nconfusion_matrix:\n",confusion_matrix(y_test,y_pred))

import joblib

joblib.dump(forest_model,"student_score_model.pkl")
joblib.dump(scaler,"student_scaler.pkl")

model=joblib.load("student_score_model.pkl")
scaler=joblib.load("student_scaler.pkl")

print("model save succefully....")