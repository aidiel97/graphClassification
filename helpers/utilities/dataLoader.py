import pandas as pd
import numpy as np

from helpers.utilities.watcher import *
from helpers.common.main import *

def rawCsv(fileName):
  ctx=fileName+' Dataset Loader'
  start = watcherStart(ctx)

  raw_df=pd.read_csv(fileName)
  watcherEnd(ctx, start)
  return raw_df

def binetflow(dataset, scenario, stringDataset=''):
  ctx=stringDataset.upper()+' '+scenario+' Dataset Loader'
  start = watcherStart(ctx)

  fileName = dataset[scenario] #load dataset
  raw_df=pd.read_csv(fileName)

  watcherEnd(ctx, start)
  return raw_df

def splitDataFrameWithProportion(dataFrame, trainProportion=defaultTrainProportion):
  ctx='Split Data Frame With Proportion'
  start= watcherStart(ctx)

  normal_df=dataFrame[dataFrame['ActivityLabel'].isin([0])] #create new normal custom dataframe
  bot_df=dataFrame[dataFrame['ActivityLabel'].isin([1])] #create a new data frame for bots

  msk_normal = np.random.rand(len(normal_df)) < trainProportion #get random 20% from normal
  msk_bot = np.random.rand(len(bot_df)) < trainProportion #get random 20% from bot

  #split normal dataset
  normal_dfTrain = normal_df[msk_normal]
  normal_dfTest = normal_df[~msk_normal]

  #split normal dataset
  bot_dfTrain = bot_df[msk_bot]
  bot_dfTest = bot_df[~msk_bot]
  
  #combine dataTest and dataTrain
  train = pd.concat([normal_dfTrain, bot_dfTrain])
  test = pd.concat([normal_dfTest, bot_dfTest])

  watcherEnd(ctx, start)
  return train, test

def splitDataFrameWithIndex(dataFrame, trainProportion=defaultTrainProportion):
  ctx='Split Data Frame With Index'
  start= watcherStart(ctx)

  normal_df=dataFrame[dataFrame['ActivityLabel'] == 'normal'] #create new normal custom dataframe
  bot_df=dataFrame[dataFrame['ActivityLabel'] == 'botnet'] #create a new data frame for bots
  bg_df=dataFrame[dataFrame['ActivityLabel'] == 'background'] #create a new data frame for bots

  trainPortionNormal = int(len(normal_df) * trainProportion)
  trainPortionBotnet = int(len(bot_df) * trainProportion)
  trainPortionBg = int(len(bg_df) * trainProportion)

  #split normal dataset
  normal_dfTrain = normal_df[:trainPortionNormal]
  normal_dfTest = normal_df[trainPortionNormal:]

  #split bot dataset
  bot_dfTrain = bot_df[:trainPortionBotnet]
  bot_dfTest = bot_df[trainPortionBotnet:]

  #split bg dataset
  bg_dfTrain = bg_df[:trainPortionBg]
  bg_dfTest = bg_df[trainPortionBg:]
  
  #combine dataTest and dataTrain
  train = pd.concat([normal_dfTrain, bot_dfTrain, bg_dfTrain])
  test = pd.concat([normal_dfTest, bot_dfTest, bg_dfTest])

  watcherEnd(ctx, start)
  return train, test

#only take samples for training, testing with all data
def splitTestAllDataframe(dataFrame, trainProportion=defaultTrainProportion):
  ctx='Split Test All Dataframe'
  start= watcherStart(ctx)

  normal_df=dataFrame[dataFrame['ActivityLabel'].isin([0])] #create new normal custom dataframe
  bot_df=dataFrame[dataFrame['ActivityLabel'].isin([1])] #create a new data frame for bots

  msk_normal = np.random.rand(len(normal_df)) < trainProportion #get random 20% from normal
  msk_bot = np.random.rand(len(bot_df)) < trainProportion #get random 20% from bot

  #split normal dataset
  normal_dfTrain = normal_df[msk_normal]

  #split normal dataset
  bot_dfTrain = bot_df[msk_bot]
  
  #combine dataTest and dataTrain
  train = pd.concat([normal_dfTrain, bot_dfTrain])
  test = dataFrame

  watcherEnd(ctx, start)
  return train, test
