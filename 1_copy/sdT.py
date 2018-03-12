import csv
import numpy as np
from sklearn import tree
with open('train.csv') as csvfile:
	data = csv.reader(csvfile,delimiter=',')    #reading data from csv file 
	train_data = []						#for features
	train_target = []						#for target variables
	for row in data:
		train_data.append(row[:20])
		train_target.append(row[20])
	test_idx = [0]
	train_data = np.delete(train_data,test_idx,axis=0)
	train_target = np.delete(train_target,test_idx)
	

	with open('test.csv') as Testdata:
			Test_data = csv.reader(Testdata,delimiter=',')
			Test_Data = []
			for Row in Test_data :
				Test_Data.append(Row)
			Test_Data = np.delete(Test_Data,test_idx,axis=0)
			#highest accuracy (svm classifier :: robust against outliers)
			"""			
			from sklearn.ensemble import RandomForestClassifier, VotingClassifier
			clfa = RandomForestClassifier(n_estimators=350)
			clfa.fit(train_data,train_target)
			pred1 = clfa.predict(Test_Data)
			"""
			from sklearn.neighbors import KNeighborsClassifier
			clf  = KNeighborsClassifier(n_neighbors=300)
			clf.fit(train_data,train_target)
			pred=clf.predict(Test_Data)			
						
			#from sklearn import svm
						
			#my_classifier=svm.SVC(kernel='linear',C=2.0,random_state=43)
			#my_classifier.fit(train_data,train_target)
			#eclf1 = VotingClassifier(estimators=[('rf', clfa), ('knn', clf), ('svc', my_classifier)], voting='soft', weights[1,1,2],flatten_transform=True)
			#eclf1=eclf1.fit(train_data,train_target)
			prediction=clf.predict(Test_Data)
			#prediction=my_classifier.predict(Test_Data)
			#Result=my_classifier.predict(Test_Data)
			#import pandas as pd 
			#df = pd.DataFrame(Result)
			#df.to_csv("file_path.csv")
			with open('resultval.csv', "w") as f :
				writer = csv.writer(f)
				ps = ['id','price_range']
				writer.writerow(ps)
				x=1
				for row in prediction:
					writer.writerow([x,row])
					x += 1
	


		
