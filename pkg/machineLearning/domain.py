import pkg.preProcessing.transform as transform
import pkg.preProcessing.handlingNull as null
import pkg.machineLearning.machineLearning as ml

from helpers.utilities.watcher import *
from helpers.utilities.dirManagement import checkDir
from helpers.common.main import *
from helpers.common.globalConfig import OUT_DIR, ABLATION

import pandas as pd
import os
import math

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
  ctx = ctx+' '+graphDetail+'-degree'
  watcherEnd(ctx, start, True)
  return df[['Address', 'Prediction']].copy()

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
  ctx = 'Modelling with '+ algorithm + ' algorithm ' + graphDetail
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

def trainingInGraph():
  arrayDfIn = []
  ##### loop all dataset (csv)
  # Specify the directory path
  directory_path = OUT_DIR+'extract/train/'

  # Get all file names in the directory
  file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

  # Print the file names
  print("File names in the directory:")
  for file_name in file_names:
    if '-in.csv' in  file_name:
      arrayDfIn.append(pd.read_csv(directory_path+file_name))
      print("Load " + file_name + "....")

  dfIn = pd.concat(arrayDfIn, axis=0)
  dfIn['ActivityLabel'] = dfIn['Label'].apply(lambda x: 1 if x == 'botnet' else 0)
  dfIn.reset_index(drop=True, inplace=True)
  
  for algo in list(ml.algorithmDict.keys()):
    modelling(dfIn, 'in', algo)


def trainingOutGraph():
  arrayDfOut = []
  ##### loop all dataset (csv)
  # Specify the directory path
  directory_path = OUT_DIR+'extract/train/'

  # Get all file names in the directory
  file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

  # Print the file names
  print("File names in the directory:")
  for file_name in file_names:
    if '-out.csv' in file_name:
      arrayDfOut.append(pd.read_csv(directory_path+file_name))
      print("Load " + file_name + "....")

  dfOut = pd.concat(arrayDfOut, axis=0)
  dfOut['ActivityLabel'] = dfOut['Label'].apply(lambda x: 1 if x == 'botnet' else 0)
  dfOut.reset_index(drop=True, inplace=True)

  for algo in list(ml.algorithmDict.keys()):
    modelling(dfOut, 'out', algo)
  
  del dfOut #flush unused dataframe

def executeAllDataGraph():
  ctx='Graph Classification - Execute All Data'
  start = watcherStart(ctx)
  ##### loop all dataset (csv)
  # Specify the directory path
  directory_path = OUT_DIR+'extract/test/'

  # Get all file names in the directory
  file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

  checkDir(OUT_DIR+'prediction/')
  # Print the file names
  generalCtx = 'graph-'
  for algo in list(ml.algorithmDict.keys()):
    for file_name in file_names:
      df = pd.read_csv(directory_path+file_name)
      df['ActivityLabel'] = df['Label'].apply(lambda x: 1 if x == 'botnet' else 0)
      df.reset_index(drop=True, inplace=True)
      
      file_name = file_name.replace("-test","")
      predictCtx = generalCtx + algo + '-' + file_name.replace(".csv","")
      if 'in' in  file_name:
        indf = predict(predictCtx, df, 'in', algo)
        result = indf.groupby('Address')['Prediction'].agg(['sum', 'count']).reset_index()
        result = result.rename(columns={'Address': 'address','Prediction': 'prediction'})
        result['Ratio'] = result['sum'] / result['count']
        result['Label'] = result['address'].apply(lambda x: 1 if x in botIP else 0)
        result.reset_index(drop=True, inplace=True)
        result.to_csv(OUT_DIR+'prediction/'+predictCtx+'.csv', index=False)
      elif 'out' in file_name:
        outdf = predict(predictCtx, df, 'out', algo)
        result = outdf.groupby('Address')['Prediction'].agg(['sum', 'count']).reset_index()
        result = result.rename(columns={'Address': 'address','Prediction': 'prediction'})
        result['Ratio'] = result['sum'] / result['count']
        result['Label'] = result['address'].apply(lambda x: 1 if x in botIP else 0)
        result.reset_index(drop=True, inplace=True)
        result.to_csv(OUT_DIR+'prediction/'+predictCtx+'.csv', index=False)

  ##### loop all dataset

  watcherEnd(ctx, start)


def combinePredictionResult():
  ctx='Graph Classification - Combine Prediction Result'
  start = watcherStart(ctx)
  ##### loop all dataset (csv)
  # Specify the directory path
  directory_path = OUT_DIR+'prediction/'
  directory_path_result = OUT_DIR+'result/'

  # Get all file names in the directory
  file_names = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
  tempFileName = ''

  weightDict = {
  'decisionTree': (0.12325019258744743,0.5393015423920258),
  'logisticRegression' : (0,0.00290217729076697),
  'naiveBayes': (0.9981427481202437,0.035289364485364384),
  'adaboost': (0.03241604539701408,0.3807663940535213),
  'extraTree': (0.05764893403874094,0.7866030186701021),
  'xGBoost': (0.05104214712892774,0.25962799183738416),
  'randomForest': (0.15037938793782574,0.7459991832477354),
  'knn': (0.03895095341457135,0.6862854309400126),
  }
  
  win = 1
  wout = 1
  algName = ''
  threshold = 40
  for file_name in file_names:
    i_start = watcherStart(ctx)
    thisDf = pd.read_csv(directory_path+file_name)

    file_name = file_name.replace(".csv","")
    file_name = file_name.replace("graph-","")
    file_name_component = file_name.split('-')
    stringDatasetName = file_name_component[1]
    stringSubDatasetName = file_name_component[2]
    coreName = '-'.join(file_name_component[:3])
    
    if (ABLATION == False and algName != file_name_component[0]):
      algName = file_name_component[0]
      win = weightDict[algName][0]*100
      wout = weightDict[algName][1]*100
      gcd_value = math.gcd(int(win), int(wout))
      if gcd_value != 0:
        win = win // gcd_value
        wout = wout // gcd_value
      else:
        win = 1
        wout = 1

    if tempFileName == coreName:
      print('Combining Prediction Result: '+coreName)
      merged_df = pd.merge(thisDf, lastDf, on='address', how='outer', suffixes=('-out', '-in'))

      merged_df['sum'] = merged_df['sum-out']*wout + merged_df['sum-in']*win
      merged_df['count'] = merged_df['count-out']*wout + merged_df['count-in']*win
      merged_df['Ratio'] = merged_df['sum'] / merged_df['count']

      merged_df['Predict'] = (merged_df['Ratio'] > threshold/100).astype(int)
      listBotnetAddress = detailBotCount[stringDatasetName][stringSubDatasetName]
      merged_df['Label'] = merged_df['address'].apply(lambda x: 1 if x in listBotnetAddress else 0)

      ml.evaluation(ctx, merged_df['Label'], merged_df['Predict'], 'COMBINED-PREDICTION-'+coreName+'th'+str(threshold)+'wout'+str(wout)+'win'+str(win))
      merged_df.reset_index(drop=True, inplace=True)
      checkDir(directory_path_result)
      merged_df.to_csv(directory_path_result+coreName+".csv")
      watcherEnd(ctx+"-"+stringDatasetName+"-"+stringSubDatasetName, i_start, True)

    lastDf = thisDf
    tempFileName= coreName

  watcherEnd(ctx, start)