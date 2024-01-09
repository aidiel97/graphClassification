import helpers.utilities.dataLoader as loader
import pkg.preProcessing.transform as preProcessing

from helpers.utilities.watcher import *
from helpers.common.main import *
from helpers.utilities.dirManagement import *
from helpers.utilities.csvGenerator import *

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
  plt.savefig('collections/'+feature+'-botnet-boxplot.png')
  botnetDf.describe().transpose().to_csv('collections/'+feature+'-botnet-describe.csv')

  plt.figure()
  backgroundEquate = equateListLength(backgroundDiff)
  backgroundDf = pd.DataFrame(backgroundEquate)
  backgroundDf.boxplot(showfliers=False)
  plt.xticks(rotation=90)
  plt.subplots_adjust(bottom=0.25)
  plt.savefig('collections/'+feature+'-background-boxplot.png')
  backgroundDf.describe().transpose().to_csv('collections/'+feature+'-background-describe.csv')

  plt.figure()
  normalEquate = equateListLength(normalDiff)
  normalDf = pd.DataFrame(normalEquate)
  normalDf.boxplot(showfliers=False)
  plt.xticks(rotation=90)
  plt.subplots_adjust(bottom=0.25)
  plt.savefig('collections/'+feature+'-normal-boxplot.png')
  normalDf.describe().transpose().to_csv('collections/'+feature+'-normal-describe.csv')

def flow(datasetName, stringDatasetName, shortName, selected, feature):
  ctx=feature+' Analysis with Statistical Approach'
  start = watcherStart(ctx)
  sequenceOf = 'SrcAddr'

  df = loader.binetflow(datasetName, selected, stringDatasetName)
  df['ActivityLabel'] = df['Label'].apply(preProcessing.labelSimplier)
  df['Unix'] = df['StartTime'].apply(preProcessing.timeToUnix).fillna(0)
  df = df.sort_values(by=[sequenceOf, 'StartTime', 'ActivityLabel'])
  df['Diff'] = df['Unix'].diff().apply(lambda x: x if x >= 0 else None)

  train, test = loader.splitDataFrameWithIndex(df)
  train.to_csv('collections/split/'+shortName+'-'+selected+'-train.csv', index=False, header=False)
  test.to_csv('collections/split/'+shortName+'-'+selected+'-test.csv', index=False, header=False)

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

  watcherEnd(ctx, start)

def equateListLength(dct):
  max_length = max(len(lst) for lst in dct.values())
  for key in dct:
      lst = dct[key]
      if len(lst) < max_length:
          lst += [np.nan] * (max_length - len(lst))
  
  return dct
