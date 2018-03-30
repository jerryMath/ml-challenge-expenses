import pandas as pd
import knn
import dataPreprocess
import calDis
from sklearn.cluster import KMeans
pd.set_option('mode.chained_assignment', None)

# .csv files of the data
training_data_file = "training_data_example.csv"
validation_data_file = "validation_data_example.csv"
employee_file = "employee.csv"


# read and process the data
trainData, validationData, trainLabel, validationLabel = dataPreprocess.question1()


# Question 1: apply KNN to do classification
correct = 0
for index, inputs in enumerate(trainData):
    prediction = knn.classifier(inputs, trainData, trainLabel, 1)
    if prediction == trainLabel[index]:
        correct += 1    
print("training accuracy is :", correct/(index + 1))   

correct = 0     
for index, inputs in enumerate(validationData):
    prediction = knn.classifier(inputs, trainData, trainLabel, 1)
    if prediction == validationLabel[index]:
        correct += 1
print("validation accuracy is :", correct/(index + 1))

# Question 2: apply K-cluster to do classification
trainData2, validationData2 = dataPreprocess.question2()
model = KMeans(n_clusters=2).fit(trainData2) 
centers = model.cluster_centers_  

businessOrPersonal = []
for ele in validationData2:
    dis1 = calDis.dis(ele, centers[0,:])
    dis2 = calDis.dis(ele, centers[1,:])
    if dis1 > dis2:
        businessOrPersonal.append('business')
    else:
        businessOrPersonal.append('personal')
        
valiDataWithNewColum = dataPreprocess.loadCsvData(validation_data_file)
valiDataWithNewColum['business or personal'] = businessOrPersonal
valiDataWithNewColum.to_csv('question2')