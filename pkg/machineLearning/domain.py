import pkg.preProcessing.transform as transform
import pkg.preProcessing.handlingNull as null
import pkg.machineLearning.machineLearning as ml

from helpers.utilities.watcher import *
from helpers.utilities.dirManagement import checkDir
from helpers.common.main import *
from helpers.common.globalConfig import OUT_DIR

import pandas as pd
import os

def preProcessingModule(df):
  #make new label for background prediciton(1/0)
  df['ActivityLabel'] = df['Label'].str.contains('botnet', case=False, regex=True).astype(int)
  #make new label for background prediciton(1/0)

  #transform with dictionary
  df['State']= df['State'].map(stateDict).fillna(0.0).astype(int)
  df['Proto']= df['Proto'].map(protoDict).fillna(0.0).astype(int)
  #transform with dictionary

  df['StartTime'] = df['StartTime'].apply(transform.timeToUnix).fillna(0)

  df['Sport'] = pd.factorize(df.Sport)[0]
  df['Dport'] = pd.factorize(df.Dport)[0]

  #transform ip to integer
  df.dropna(subset = ["DstAddr"], inplace=True)
  df.dropna(subset = ["SrcAddr"], inplace=True)
  df['SrcAddr'] = df['SrcAddr'].apply(transform.ipToInteger).fillna(0)
  df['DstAddr'] = df['DstAddr'].apply(transform.ipToInteger).fillna(0)
  #transform ip to integer

  null.setEmptyString(df)
  # cleansing.featureDropping(df, ['sTos','dTos','Dir'])

  #one hot encode
  dir_values_to_encode = ['  <->','   ->','  who','  <-','  <?>','   ?>','  <?']
  dummy_cols =pd.get_dummies(
    df['Dir'].apply(lambda x: x if x in dir_values_to_encode else 'other'), columns=dir_values_to_encode, prefix='Dir')
  df = pd.concat([df,dummy_cols],axis=1)
  df.drop(columns='Dir', axis=1, inplace=True)
  
  return df

def predict(ctx, df, graphDetail, algorithm='randomForest'):
  start = watcherStart(ctx)

  df.fillna(0, inplace=True)
  categorical_features=[feature for feature in df.columns if (
    df[feature].dtypes=='O' or feature =='SensorId' or feature =='ActivityLabel'
  )]
  x = df.drop(categorical_features,axis=1)
  y = df['ActivityLabel']
  predictionResult = ml.classification(x, graphDetail, algorithm)
  ml.evaluation(ctx, y, predictionResult, algorithm)
  
  df['Prediction'] = predictionResult
  checkDir(OUT_DIR+'prediction/'+graphDetail+'/')
  df.to_csv(OUT_DIR+'prediction/'+graphDetail+'/'+ctx+'.csv', index=False)

  ctx = ctx+' '+graphDetail+'-degree'
  watcherEnd(ctx, start, True)
  return predictionResult

def methodEvaluation(dataset, actual_df, predicted_df, method='Proposed Sequence Pattern Miner'):
  ctx='Method Evaluation'
  start = watcherStart(ctx)
  addressPredictedAsBotnet = predicted_df['SrcAddr'].unique()

  actual_df['ActualClass'] = actual_df['Label'].str.contains('botnet', case=False, regex=True)
  result_df = actual_df.groupby('SrcAddr')['ActualClass'].apply(lambda x: x.mode()[0]).reset_index()
  result_df.columns = ['SrcAddr','ActualClass']
  result_df['PredictedClass'] = result_df['SrcAddr'].isin(addressPredictedAsBotnet)

  ml.evaluation(dataset, result_df['ActualClass'], result_df['PredictedClass'], method)
  watcherEnd(ctx, start)

def modelling(df, graphDetail, algorithm='randomForest'):
  ctx = 'Modelling with '+ algorithm + ' algorithm'
  start = watcherStart(ctx)

  df.fillna(0, inplace=True)
  categorical_features=[feature for feature in df.columns if (
    df[feature].dtypes=='O' or feature =='SensorId' or feature =='ActivityLabel'
  )]
  x = df.drop(categorical_features,axis=1)
  y = df['ActivityLabel']
  ml.modelling(x, y, graphDetail, algorithm)
  #modelling

  watcherEnd(ctx, start, True)

# def executeAllData():
#   ctx='Machine learning Classification - Execute All Data'
#   start = watcherStart(ctx)

#   for algo in list(ml.algorithmDict.keys()):
#     modellingWithCTU(algo)
#   ##### loop all dataset
#     for dataset in listAvailableDatasets[:3]:
#       print('\n'+dataset['name'])
#       for scenario in dataset['list']:
#         print(scenario)
#         datasetDetail={
#           'datasetName': dataset['list'],
#           'stringDatasetName': dataset['name'],
#           'selected': scenario
#         }

#         raw_df = loader.binetflow(
#           datasetDetail['datasetName'],
#           datasetDetail['selected'],
#           datasetDetail['stringDatasetName'])

#         df = raw_df.copy() #get a copy from dataset to prevent processed data
#         result = predict(df, algo)
#         raw_df['predictionResult'] = result
#         new_df = raw_df[raw_df['predictionResult'] == 1]
        
#         datasetName = datasetDetail['stringDatasetName']+'-'+datasetDetail['selected']
#         methodEvaluation(datasetName, raw_df, new_df, algo)
#   ##### loop all dataset

#   watcherEnd(ctx, start)

def trainingAllAlgorithm():
  arrayDfIn = []
  arrayDfOut = []
  ##### loop all dataset (csv)
  # Specify the directory path
  directory_path = OUT_DIR+'extract/'

  # Get all file names in the directory
  file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

  # Print the file names
  print("File names in the directory:")
  for file_name in file_names:
    if '-train' in file_name:
      if '-in.csv' in  file_name:
        arrayDfIn.append(pd.read_csv(directory_path+file_name))
      elif '-out.csv' in file_name:
        arrayDfOut.append(pd.read_csv(directory_path+file_name))

  dfIn = pd.concat(arrayDfIn, axis=0)
  dfIn['ActivityLabel'] = dfIn['Label'].str.contains('botnet', case=False, regex=True).astype(int)
  dfIn.reset_index(drop=True, inplace=True)
  
  dfOut = pd.concat(arrayDfOut, axis=0)
  dfOut['ActivityLabel'] = dfOut['Label'].str.contains('botnet', case=False, regex=True).astype(int)
  dfOut.reset_index(drop=True, inplace=True)

  for algo in list(ml.algorithmDict.keys()):
    modelling(dfIn, 'in', algo)
    modelling(dfOut, 'out', algo)

def executeAllDataGraph():
  ctx='Graph Classification - Execute All Data'
  start = watcherStart(ctx)
  ##### loop all dataset (csv)
  # Specify the directory path
  directory_path = OUT_DIR+'extract/'

  # Get all file names in the directory
  file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

  # Print the file names
  generalCtx = 'graph-'
  for algo in list(ml.algorithmDict.keys()):
    for file_name in file_names:
      if 'test' in file_name:
        df = pd.read_csv(directory_path+file_name)
        df['ActivityLabel'] = df['Label'].str.contains('botnet', case=False, regex=True).astype(int)
        df.reset_index(drop=True, inplace=True)
        
        file_name = file_name.replace("-test","")
        predictCtx = generalCtx + algo + '-' + file_name.replace(".csv","")
        if 'in' in  file_name:
          predict(predictCtx, df, 'in', algo)
        elif 'out' in file_name:
          predict(predictCtx, df, 'out', algo)
  ##### loop all dataset

  watcherEnd(ctx, start)
