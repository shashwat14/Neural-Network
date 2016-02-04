# Neural-Network
#A Feed Forward Neural Network class with user defined layers and neuron numbers. Assumes single output neuron and uses backpropagation algorithm

#A sample python program for training the network

import numpy as np
from NeuralNetwork import NeuralNetwork
import openpyxl
from openpyxl import load_workbook

def getData():
    InputLayer =[]
    dataColumn = []
    wb = load_workbook(filename = 'pfdb.xlsx');
    sheet = wb.get_active_sheet()
    InputLayerLength =  len(sheet.rows[0])
    for i in range(0,InputLayerLength):
        for data in sheet.columns[i]:
            dataColumn.append(float(data.value))
        dataColumn = Normalize(dataColumn)
        length = len(dataColumn)
        InputLayer.append(dataColumn)
        dataColumn = []
    return InputLayer,length

def Normalize(dataColumn):
    #print dataColumn
    newDataColumn = []
    maximum = max(dataColumn)
    minimum = min(dataColumn)
    for each in dataColumn: 
        norm = (each - minimum)/(maximum-minimum)
        newDataColumn.append(norm)
        #print norm
    return newDataColumn


InputArray,length = getData()
OutputLayer = np.array([InputArray[1]])
ones = [1]*length
#InputLayer = np.array([InputArray[0],InputArray[2],InputArray[3],ones])
InputLayer = np.array([InputArray[0],ones])
InputLayer = InputLayer.T
y = OutputLayer.T

nn = NeuralNetwork(4,4)
nn.declareInput(InputLayer)
nn.declareTarget(y)
nn.train(1000000)
