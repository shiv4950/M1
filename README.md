import numpy as np
from sklearn.model_selection import KFold,cross_val_score


from sklearn.datasets import load_iris  # Example dataset
from sklearn.svm import SVC
data = load_iris()
X = data.data  # Features
y = data.target  # Labels
svm_classifier=SVC(kernel='linear')
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)
cross_val_results=cross_val_score(svm_classifier,X,y,cv=kf)


print("cross validation score",cross_val_results)
mean_acc=cross_val_results.mean()
print("mean_accuracy",mean_acc)

***#decision tree 
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score


data = load_iris()
x=data.data
y=data.target
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)


plt.figure(figsize=(12, 8))
tree.plot_tree(clf, filled=True, feature_names=data.data , class_names=data.target_names)
plt.show()



***bagging-
from sklearn.ensemble import BaggingClassifier
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas  as pd
from sklearn.ensemble import BaggingClassifier
# Load the dataset
data=load_iris()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
base_classifier = DecisionTreeClassifier(random_state=42)
bagging_classifier = BaggingClassifier(base_classifier, n_estimators=10,random_state=42)
bagging_classifier.fit(X_train, y_train)
y_pred = bagging_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

***lineae regreesion-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split


data=pd.read_csv("salary_data.csv")


x = data[["YearsExperience"]]
y = data["Salary"]


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)


model = LinearRegression()
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
print(y_pred)


# for training
plt.scatter(x_train,y_train)
plt.plot(x_train,model.predict(x_train))
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('for trainingn')
plt.show()


#for testing
plt.scatter(x_test,y_test)
plt.plot(x_test,y_pred, color='red', linewidth=2)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('for testing')
plt.show()

***polynomial regression-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv("Position_Salaries.csv")
X = data["Level"]
y = data["Salary"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  # Set random state for reproducibility
poly = PolynomialFeatures(degree=4)
X_train_poly = poly.fit_transform(X_train.values.reshape(-1, 1))
X_test_poly = poly.transform(X_test.values.reshape(-1, 1))
model = LinearRegression()
model.fit(X_train_poly, y_train)


y_pred = model.predict(X_test_poly)
print(y_pred)


X_plot = np.linspace(X.min(), X.max(), 100)  # Create a range of points for plotting the curve
X_plot_poly = poly.transform(X_plot.reshape(-1, 1))  # Transform for polynomial features


# Training set


plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_plot, model.predict(X_plot_poly), color='red', label='Polynomial Regression Curve (Training)')
plt.xlabel('Position Levels')
plt.ylabel('Salary')
plt.title('Polynomial Regression Model..for trainning data')
plt.show()


# Testing set
plt.scatter(X_test, y_test, color='blue', label='Testing Data')
plt.plot(X_plot, model.predict(X_plot_poly), color='red', label='Polynomial Regression Curve (Testing)')
plt.xlabel('Position Levels')
plt.ylabel('Salary')
plt.title('Polynomial Regression Model for testing data ')
plt.show()


# Predict on entire dataset
X_all_poly = poly.transform(X.values.reshape(-1, 1))
y_pred_all = model.predict(X_all_poly)
print("Predicted Salaries (Full Dataset):")
print(pd.DataFrame({'Position Levels': X, 'Predicted Salary': y_pred_all}))


***logistic regression-
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt


a=['User_ID','Gender','Age','EstimatedSalary','Purchased']


data = pd.read_csv("User_Data.csv")
X = data[['Age']]
y = data[["Purchased"]]


X_train, X_test,y_train, y_test = train_test_split(X, y,test_size=0.20, random_state=23)


clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Logistic Regression model accuracy (in %):", acc*100)


#for training
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred, color='red', linewidth=2)
plt.xlabel('age')
plt.ylabel('gender')
plt.title('for testing')
plt.show()


#for testing
plt.scatter(X_test,y_test)
plt.plot(X_test,y_pred, color='red', linewidth=2)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.title('for testing')
plt.show()

