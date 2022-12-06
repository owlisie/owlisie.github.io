import os
import sys
import csv
import numpy as np
import pandas as pd
import torch

dataPath = './mamen/'
dirList = os.listdir(dataPath)
folList = []

folList = ['dataset1', 'dataset2', 'dataset3']
"""
for dir in dirList:
    if os.listdir(dataPath + dir + '/'):
        folList.append(dir)
"""
for fol in folList:
    fileList = os.listdir(dataPath + fol + '/')
    os.makedirs(dataPath + 'CSV' + fol + '/', exist_ok=True)
    for f in fileList:
        fileName = dataPath + fol + '/' + f
        savefileName = dataPath + 'CSV' + fol + '/' + f.split('.')[0] + '.csv'
        fileCSV = open(savefileName, 'w', newline='')
        fileCSV = csv.writer(fileCSV)
        with open(fileName, 'rt', encoding='UTF8') as file:
            dataLines = file.readlines()

        for line in dataLines:
            line = line.strip()
            fileCSV.writerow(line)
