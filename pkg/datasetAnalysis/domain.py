import helpers.utilities.dataLoader as loader
import pkg.preProcessing.transform as preProcessing

from helpers.utilities.watcher import *
from helpers.common.main import *
from helpers.utilities.dirManagement import *
from helpers.utilities.csvGenerator import *
from helpers.utilities.dirManagement import checkDir

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

botnetDiff = {}
backgroundDiff = {}
normalDiff = {}

def splitAndTimeGap(): main('Diff')
def SrcBytes(): main('SrcBytes')

def main(feature):
  for dataset in listAvailableDatasets[:3]:
    print('\n'+dataset['name'])
    for scenario in dataset['list']:
      print(scenario)
      dataset = {
        'list': dataset['list'],
        'name': dataset['name'],
        'shortName': dataset['shortName']
      }
      flow(dataset['list'], dataset['name'], dataset['shortName'], scenario, feature)

  # Create a boxplot
  plt.figure()
  botnetEquate = equateListLength(botnetDiff)
  botnetDf = pd.DataFrame(botnetEquate)
  botnetDf.boxplot(showfliers=False)
  plt.xticks(rotation=90)
  plt.subplots_adjust(bottom=0.25)
  plt.savefig(OUT_DIR+''+feature+'-botnet-boxplot.png')
  botnetDf.describe().transpose().to_csv(OUT_DIR+''+feature+'-botnet-describe.csv')

  plt.figure()
  backgroundEquate = equateListLength(backgroundDiff)
  backgroundDf = pd.DataFrame(backgroundEquate)
  backgroundDf.boxplot(showfliers=False)
  plt.xticks(rotation=90)
  plt.subplots_adjust(bottom=0.25)
  plt.savefig(OUT_DIR+''+feature+'-background-boxplot.png')
  backgroundDf.describe().transpose().to_csv(OUT_DIR+''+feature+'-background-describe.csv')

  plt.figure()
  normalEquate = equateListLength(normalDiff)
  normalDf = pd.DataFrame(normalEquate)
  normalDf.boxplot(showfliers=False)
  plt.xticks(rotation=90)
  plt.subplots_adjust(bottom=0.25)
  plt.savefig(OUT_DIR+''+feature+'-normal-boxplot.png')
  normalDf.describe().transpose().to_csv(OUT_DIR+''+feature+'-normal-describe.csv')

def flow(datasetName, stringDatasetName, shortName, selected, feature):
  ctx=feature+' Analysis with statistical approach '+shortName+'-'+selected
  start = watcherStart(ctx)
  sequenceOf = 'SrcAddr'
  
  checkDir(OUT_DIR+'split/')
  df = loader.binetflow(datasetName, selected, stringDatasetName)
  df['ActivityLabel'] = df['Label'].apply(preProcessing.labelSimplier)
  df['Unix'] = df['StartTime'].apply(preProcessing.timeToUnix).fillna(0)

  if stringDatasetName == 'ctu' and selected == 'scenario7':
    train=df
    test=df
  else:
    train, test = loader.splitDataFrameWithIndex(df)

  # generate diff feature on test data
  train = train.sort_values(by=[sequenceOf, 'Unix'])
  train = preProcessing.calculate_diff(train)
  
  # generate diff feature on test data
  test = test.sort_values(by=[sequenceOf, 'Unix'])
  test = preProcessing.calculate_diff(test)

  # export the data
  checkDir(OUT_DIR+'split/train/')
  checkDir(OUT_DIR+'split/test/')
  if stringDatasetName != 'ctu' and selected != 'scenario7':
    train.to_csv(OUT_DIR+'split/train/'+shortName+'-'+selected+'.csv', index=False, header=True)
  
  test.to_csv(OUT_DIR+'split/test/'+shortName+'-'+selected+'.csv', index=False, header=True)

  botnet = train[train['ActivityLabel'] == 'botnet']
  normal = train[train['ActivityLabel'] == 'normal']
  background = train[train['ActivityLabel'] == 'background']

  tgBotnet = botnet.loc[botnet[feature] != 0, feature].values.tolist()
  tgBackground = background.loc[background[feature] != 0, feature].values.tolist()
  tgNormal = normal.loc[normal[feature] != 0, feature].values.tolist()

  datasetVariableName = shortName+'('+selected[8:]+')'
  botnetDiff[datasetVariableName] = tgBotnet
  backgroundDiff[datasetVariableName] = tgBackground
  normalDiff[datasetVariableName] = tgNormal

  watcherEnd(ctx, start, True)

def equateListLength(dct):
  max_length = max(len(lst) for lst in dct.values())
  for key in dct:
      lst = dct[key]
      if len(lst) < max_length:
          lst += [np.nan] * (max_length - len(lst))
  
  return dct
