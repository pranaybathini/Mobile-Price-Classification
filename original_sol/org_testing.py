import csv
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import accuracy_score    #accuaracy_score for classification

#matplotlib  for plotting and to visualize data
import matplotlib.pyplot as plt  

with open('train.csv') as csvfile:					#opening train.csv
	data = pd.read_csv(csvfile,delimiter=',')		#loading train.csv as pandas dataframe
	
	y = data['price_range']					#y is my target vector
	del data['price_range']						#deleting target vector from dataframe
	X = data									#X is my feature vector
	 
	 #splitting X & y for training and testing
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,random_state=42)
	
	
	#1.Decision Tree Classifier    	#best_score:86.2
	clfa = tree.DecisionTreeClassifier(max_depth=15)
	clfa.fit(X_train,y_train)
	predict1 = clfa.predict(X_test)
	print("Decision Tree     ",accuracy_score(y_test,predict1)*100)
	
	
	#2.Logistic regression			#77.25
	from sklearn.linear_model import LogisticRegression
	clfb  = LogisticRegression()
	clfb.fit(X_train,y_train)
	predict2 = clfb.predict(X_test)
	print("Logistic regression ",accuracy_score(y_test,predict2)*100)
	
	
	#3.KnearestNeighbors  #best_score : 93.2
	from sklearn.neighbors import KNeighborsClassifier
	clfc  = KNeighborsClassifier(n_neighbors=49)
	clfc.fit(X_train,y_train)
	predict3 = clfc.predict(X_test)
	print("KnearestNeighbors ",accuracy_score(y_test,predict3)*100)
	
	
	
	#4.svm      #best_score : 96.5
	from sklearn import svm
	clfd=svm.SVC(C=1.5,kernel='linear')
	clfd.fit(X_train,y_train)
	predict4 = clfd.predict(X_test)
	print(predict4)
	print("SVM Classifier ",accuracy_score(y_test,predict4)*100)
	
	
	#5.RandomForestClassifier   #best_score : 87.2
	from sklearn.ensemble import RandomForestClassifier
	clfe = RandomForestClassifier(n_estimators=300)
	clfe.fit(X_train,y_train)
	predict5 = clfe.predict(X_test)
	print("RandomForest ",accuracy_score(y_test,predict5)*100)
	
	
	

	from xgboost import XGBClassifier
	clfh = XGBClassifier()
	clfh.fit(X_train,y_train)
	predict7 = clfh.predict(X_test)
	
	
	#plotting features importance
	plt.figure(figsize=(20,15))	
	#xgb.plot_importance(clfh, ax=plt.gca())
	#plt.show()
	#plot tree
	xgb.plot_tree(clfh, ax=plt.gca())
	plt.show()
	xgb.to_graphviz(clfh, fmap='', num_trees=0, rankdir='UT', yes_color='#0000FF', no_color='#FF0000')
	print("Model Accuray: {:.2f}%".format(100*clfh.score(X_test, y_test)))
	print("XGB ",accuracy_score(y_test,predict7)*100)
	print("Number of boosting trees: {}".format(clfh.n_estimators))
	print("Max depth of trees: {}".format(clfh.max_depth))
	print("Objective function: {}".format(clfh.objective))
	 
	
	
	#8.Gaussian naive bayes
	from sklearn.naive_bayes import GaussianNB
	model = GaussianNB()
	model.fit(X_train,y_train)
	predict8 = model.predict(X_test)
	print("NB ",accuracy_score(y_test,predict8)*100)

	
	# 10.Train the logistic regeression 
	from sklearn.linear_model import LogisticRegressionCV
	clf = LogisticRegressionCV()
	clf.fit(X_train,y_train)
	predict10 = clf.predict(X_test)
	print("Logistic Regression cv ",accuracy_score(y_test,predict10)*100)
	
	#11. GradientBoostingClassifier
	from sklearn.ensemble import GradientBoostingClassifier
	gbm = GradientBoostingClassifier(max_depth=6)
	gbm.fit(X_train,y_train)
	predict11 = gbm.predict(X_test)
	print("GBM ",accuracy_score(y_test,predict11)*100)
	
	
	
