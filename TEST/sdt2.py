import csv
import numpy as np
from sklearn import tree
with open('Train.csv') as csvfile:
	data = csv.reader(csvfile,delimiter=',')    #reading data from csv file 
	train_data = []						#for features
	train_target = []						#for target variables
	for row in data:
		train_data.append(row[:20])
		train_target.append(row[20])
	test_idx = [0]
	train_data = np.delete(train_data,test_idx,axis=0)
	train_target = np.delete(train_target,test_idx)
	

	with open('Test.csv') as Testdata:
			Test_data = csv.reader(Testdata,delimiter=',')
			Test_Data = []
			for Row in Test_data :
				Test_Data.append(Row[1:])
			Test_Data = np.delete(Test_Data,test_idx,axis=0) 
			#print(Test_Data[0])
			
			#from sklearn.neighbors import KNeighborsClassifier
			#my_classifier= KNeighborsClassifier(n_neighbors=50)
			from sklearn.neural_network import MLPClassifier
			my_classifier= MLPClassifier(solver='lbfgs',hidden_layer_sizes=(100,100,100))
			my_classifier.fit(train_data,train_target)
			prediction=my_classifier.predict(Test_Data)
			Result=my_classifier.predict(Test_Data)
			
			with open('result.csv', "w") as f :
				writer = csv.writer(f)
				ps = ['id','price_range']
				writer.writerow(ps)
				
				x=1
				for row in Result:
					writer.writerow([x,row])
					x += 1
			
	


		
