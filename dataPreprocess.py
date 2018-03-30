#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 18:40:00 2018

@author: jiayicheng
"""
import pandas as pd
import numpy as np
training_data_file = "training_data_example.csv"
validation_data_file = "validation_data_example.csv"
employee_file = "employee.csv"

def loadCsvData(fileName):
    # read .csv by pandas
    data = pd.read_csv(fileName)
    return data

def convert2onehot(data):
    # covert data to onehot representation
    return pd.get_dummies(data)

def question1():
    rawTrain = loadCsvData(training_data_file)
    rawVali = loadCsvData(validation_data_file)
    rawData = pd.concat([rawTrain, rawVali])
    # feature selection
    reducedData = rawData.drop(['date','tax name','tax amount'],axis=1)
    
    labels = reducedData[['category']]
    labels = np.array(labels)
    labels = labels.tolist()
    labels =[ele[0] for ele in labels]
    
    reducedData.pop('category')
    oneHotData = convert2onehot(reducedData)
    
    #normalized the data for specifica colums
    trainData, valiData = oneHotData[0:24], oneHotData[24:]
    feature = 'pre-tax amount'
    trainData[feature] = (trainData[feature] - trainData[feature].min())/ (trainData[feature].max() - trainData[feature].min())
    valiData[feature] = (valiData[feature] - valiData[feature].min())/ (valiData[feature].max() - valiData[feature].min())
    feature = 'employee id'
    trainData[feature] = (trainData[feature] - trainData[feature].min())/ (trainData[feature].max() - trainData[feature].min())
    valiData[feature] = (valiData[feature] - valiData[feature].min())/ (valiData[feature].max() - valiData[feature].min())
    
    trainData = trainData.values.astype(np.float32)
    valiData = valiData.values.astype(np.float32)
    
    return trainData, valiData, labels[0:24], labels[24:]

def question2():
    rawTrain = loadCsvData(training_data_file)
    rawValidation = loadCsvData(validation_data_file)
    rawData = pd.concat([rawTrain, rawValidation])
    # feature selection
    reducedData = rawData.drop(['date','tax name','tax amount'],axis=1)
    oneHotData = convert2onehot(reducedData)
    
    trainData, valiData = oneHotData[0:24], oneHotData[24:]
    feature = 'pre-tax amount'
    trainData[feature] = (trainData[feature] - trainData[feature].min())/ (trainData[feature].max() - trainData[feature].min())
    valiData[feature] = (valiData[feature] - valiData[feature].min())/ (valiData[feature].max() - valiData[feature].min())
    feature = 'employee id'
    trainData[feature] = (trainData[feature] - trainData[feature].min())/ (trainData[feature].max() - trainData[feature].min())
    valiData[feature] = (valiData[feature] - valiData[feature].min())/ (valiData[feature].max() - valiData[feature].min())
    
    trainData = trainData.values.astype(np.float32)
    valiData = valiData.values.astype(np.float32)
    
    return trainData, valiData  