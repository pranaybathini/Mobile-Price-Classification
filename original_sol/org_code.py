import csv
import numpy as np
from sklearn import tree
with open('train.csv') as csvfile:
	data = csv.reader(csvfile,delimiter=',')    #reading data from csv file 
	train_data = []						#for features
	train_target = []						#for target variables
	for row in data:
		train_data.append(row[:20])		#appending features to train_data
		train_target.append(row[20])		#appending features to test_target
	test_idx = [0]						#index 0 contains headers(Strings) ,so we have to delete them
	train_data = np.delete(train_data,test_idx,axis=0)			#axis = 0 indicates axis along which we have to delete the sub arrays(2nd argument)
	train_target = np.delete(train_target,test_idx,axis=0)		#same as above
	

	with open('test.csv') as Testdata:				#with 'with' keyword ,no need to close the file separately
			Test_data = csv.reader(Testdata,delimiter=',') #csv.reader returns a reader object to iterate over all the lines 
			Test_Data = []				#for storing test_data
			for Row in Test_data :			#iterating through reader object and appending features as a list #id column ignored
				Test_Data.append(Row[1:])
			Test_Data = np.delete(Test_Data,test_idx,axis=0) #deleting feature names 
			
			
			from sklearn import svm
			my_classifier=svm.SVC(C=1.4, kernel='linear', gamma=0.1) #c penalty parameter (smaller c : large margin ,ignore outliers)
																#larger c : low margin ,sometimes may result in mislabeling
																#gamma :kernal coeff (will try to fit as per training data)
			
			my_classifier.fit(train_data,train_target)			#fitting tarining data		
			Result=my_classifier.predict(Test_Data)				#predicting data
			
			with open('result.csv', "w") as f :
				writer = csv.writer(f)				#creating a writer object 
				ps = ['id','price_range']
				writer.writerow(ps)				#writing to result.csv 
				
				x=1
				for row in Result:
					writer.writerow([x,row])
					x += 1
			
	


		
