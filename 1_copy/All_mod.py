import csv
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

with open('train.csv') as csvfile:
	data = pd.read_csv(csvfile,delimiter=',')
	
	y = data['price_range']
	del data['price_range']
	X = data
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
	
	#1.Decision Tree Classifier
	clfa = tree.DecisionTreeClassifier()
	clfa.fit(X_train,y_train)
	predict1 = clfa.predict(X_test)
	print("Decision Tree     ",accuracy_score(y_test,predict1)*100)
	
	#3.KnearestNeighbors
	from sklearn.neighbors import KNeighborsClassifier
	clfc  = KNeighborsClassifier(n_neighbors=49)
	clfc.fit(X_train,y_train)
	predict3 = clfc.predict(X_test)
	print("KnearestNeighbors ",accuracy_score(y_test,predict3)*100)
	
	
	#5.RandomForestClassifier
	from sklearn.ensemble import RandomForestClassifier
	clfe = RandomForestClassifier(n_estimators=1000)
	clfe.fit(X_train,y_train)
	predict5 = clfe.predict(X_test)
	print("RandomForest ",accuracy_score(y_test,predict5)*100)
	
	
	clfd=svm.SVC(kernel='linear',C=1)
	clfd.fit(X_train,y_train)
	predict4 = clfd.predict(X_test)
	print("SVM Classifier ",accuracy_score(y_test,predict4)*100)
	"""
	i=[0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5]
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	svc = svm.SVC()
	clf = GridSearchCV(svc, parameters)
	clf.fit(X_train,y_train)
	predict = clf.predict(X_test)
	print("SVM  ",accuracy_score(y_test,predict)*100)
	print(sorted(clf.cv_results_.keys()))
	"""
