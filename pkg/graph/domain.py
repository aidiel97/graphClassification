import networkx as nx
import pandas as pd
from tqdm import tqdm
import time
import os

import helpers.utilities.dataLoader as loader
import pkg.machineLearning.machineLearning as ml
import interfaces.cli.dataset as datasetMenu

from helpers.utilities.watcher import *
from helpers.common.main import *
from pkg.graph.models import *
from pkg.graph.generator import *
from pkg.graph.handler import *

from sklearn.preprocessing import StandardScaler 

def graphClassificationModelling():
    ctx = 'Graph based classification - Modelling'
    start = watcherStart(ctx)

    keysAlg = list(ml.algorithmDict.keys())
    print("Choose one of this algorithm to train :")
    
    i=1
    for alg in keysAlg:
        print(str(i)+". "+alg)
        i+=1
    
    indexAlg = input("Enter Menu: ")
    algorithm = keysAlg[int(indexAlg)-1]

    #modelling
    #### PRE DEFINED TRAINING DATASET FROM http://dx.doi.org/10.1016/j.cose.2014.05.011
    trainDataset = ['scenario3','scenario4','scenario5','scenario7','scenario10','scenario11','scenario12','scenario13']
    arrayDf = []
    datasetName = nccGraphCTU
    stringDatasetName = 'nccGraphCTU'
    for selected in trainDataset:
        arrayDf.append(loader.binetflow(datasetName, selected, stringDatasetName))
    df = pd.concat(arrayDf, axis=0)
    df.reset_index(drop=True, inplace=True)
    botIP = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

    df['CVReceivedBytes'] = df['CVReceivedBytes'].fillna(0)
    df['CVSentBytes'] = df['CVSentBytes'].fillna(0)
    df['ActivityLabel'] = df['Address'].isin(botIP).astype(int)
    
    categorical_features=[feature for feature in df.columns if (
        df[feature].dtypes=='O' or feature =='SensorId' or feature =='ActivityLabel'
    )]
    y = df['ActivityLabel']
    x = df.drop(categorical_features,axis=1)
    scaler = StandardScaler()
    scaler.fit(x)
    x = scaler.transform(x)

    ml.modelling(x, y, algorithm)
    
    test_df = loader.binetflow(nccGraphCTU, 'scenario9', 'nccGraphCTU')
    test_df['ActivityLabel'] = test_df['Address'].isin(botIP).astype(int)
    
    for col in protoDict.keys():
        test_df[col] = (test_df['Proto'] == col).astype(int)

    test_df['CVReceivedBytes'] = test_df['CVReceivedBytes'].fillna(0)
    test_df['CVSentBytes'] = test_df['CVSentBytes'].fillna(0)

    y_test = test_df['ActivityLabel']
    x_test = test_df.drop(categorical_features,axis=1)
    scaler = StandardScaler()
    scaler.fit(x_test)
    x_test = scaler.transform(x_test)

    predict_result = ml.classification(x_test, algorithm)
    ml.evaluation(ctx, y_test, predict_result, algorithm)

    watcherEnd(ctx, start)

def executeAllData():
  ctx='Graph based analysis - Execute All Data'
  start = watcherStart(ctx)

  ##### loop all dataset (csv)
  # Specify the directory path
  directory_path = 'collections/split/'

  # Get all file names in the directory
  file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

  # Print the file names
  print("File names in the directory:")
  for file_name in file_names:
      dftoGraph(directory_path+file_name)

  watcherEnd(ctx, start)
