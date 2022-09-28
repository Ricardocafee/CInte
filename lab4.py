
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import datetime
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier


header_list= ["Sepal length in cm","Sepal width in cm", "Petal length in cm", "Petal width in cm", "Class"]

df = pd.read_csv('iris.data', sep=',', names=header_list)

header_list2 = ["Age of patient", "Year of operation", "Number of positive axillary nodes detected", "Survival status"]

df2 = pd.read_csv('haberman.data', sep=',', names=header_list2)

aux = df.head()  #Return the first n rows (5 by default)
aux_2 = df.tail() #Return the last n rows (5 by default)
aux_3 = df.dtypes #View the data types

sns.relplot(data = df, x = "Sepal length in cm",y = "Sepal width in cm", hue="Class", size="Petal width in cm")
sns.relplot(data = df, x = "Sepal length in cm",y = "Sepal width in cm", hue="Class", size="Petal length in cm")
sns.relplot(data = df, x = "Petal length in cm",y = "Petal width in cm", hue="Class", size="Sepal width in cm")
sns.relplot(data = df, x = "Petal length in cm",y = "Petal width in cm", hue="Class", size="Sepal length in cm")

##plt.show()

sns.relplot(data = df2, x = "Age of patient",y = "Year of operation", hue= "Survival status", size="Number of positive axillary nodes detected", palette=["b","r"])

#plt.show()

X = df[{"Sepal length in cm", "Sepal width in cm", "Petal length in cm", "Petal width in cm"}].to_numpy()
y = df["Class"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

##Naives Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nNaives Bayes\n")
print("Accuracy: ", acc)
print("Precision: ",prec)
print("Recall: ", recall)
print("Confusion matrix: \n", conf_matrix)

#LinearSVC

clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nLinear SVC\n")
print("Accuracy: ", acc)
print("Precision: ",prec)
print("Recall: ", recall)
print("Confusion matrix: \n", conf_matrix)

#svm

clf = svm.SVC(kernel="linear")  #Linear Kernel
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nsvm\n")
print("Accuracy: ", acc)
print("Precision: ",prec)
print("Recall: ", recall)
print("Confusion matrix: \n", conf_matrix)

#K-Neighbors

neigh = KNeighborsClassifier()
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nK-Neighbors\n")
print("Accuracy: ", acc)
print("Precision: ",prec)
print("Recall: ", recall)
print("Confusion matrix: \n", conf_matrix)


##Second Problem (Binary problem)

print("\n=====================================================================================")
print("Haberman Problem (Binary)")
print("=====================================================================================")

X = df2[{"Age of patient", "Year of operation", "Number of positive axillary nodes detected"}].to_numpy()
y = df2["Survival status"].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 

##Naives Bayes
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nNaives Bayes\n")
print("Accuracy: ", acc)
print("Precision: ",prec)
print("Recall: ", recall)
print("Confusion matrix: \n", conf_matrix)

#LinearSVC

clf = LinearSVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nLinear SVC\n")
print("Accuracy: ", acc)
print("Precision: ",prec)
print("Recall: ", recall)
print("Confusion matrix: \n", conf_matrix)

#svm

clf = svm.SVC(kernel="linear")  #Linear Kernel
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nsvm\n")
print("Accuracy: ", acc)
print("Precision: ",prec)
print("Recall: ", recall)
print("Confusion matrix: \n", conf_matrix)

#K-Neighbors

neigh = KNeighborsClassifier()
neigh.fit(X_train, y_train)
y_pred = neigh.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nK-Neighbors\n")
print("Accuracy: ", acc)
print("Precision: ",prec)
print("Recall: ", recall)
print("Confusion matrix: \n", conf_matrix)

