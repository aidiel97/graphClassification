import re
import pandas as pd
import socket, struct
import numpy as np
import time

from datetime import datetime
from helpers.utilities.watcher import *
from helpers.common.main import *

def ipToInteger(ip):
  try:
    packedIP = socket.inet_aton(ip)
    return struct.unpack("!L", packedIP)[0]
  except OSError:
    return np.nan #return NaN when IP Address is not valid
  
def calculate_diff(df):
  # Initialize an empty list to store the calculated differences
  diff_values = []

  # Iterate over the DataFrame
  for i in range(len(df)):
      if i > 0 and df['SrcAddr'].iloc[i] == df['SrcAddr'].iloc[i - 1]:
          diff_values.append(df['Unix'].iloc[i] - df['Unix'].iloc[i - 1])
      else:
          # If the condition is not met, append None
          diff_values.append(None)

  # Create a new column 'Diff' in the DataFrame and assign the calculated differences
  df['Diff'] = diff_values

  # Return the entire DataFrame with the new 'Diff' column
  return df

def timeToUnix(startTime):
  t = pd.Timestamp(startTime)
  unix_time = time.mktime(t.timetuple())
  return unix_time

def labelSimplier(label):
  label = label.lower()
  if 'botnet' in label: 
    return 'botnet'
  elif 'background' in label:
    return 'background'
  else:
    return 'normal'

def labelProcessing(label):
  extractWordCollection = ['V','flow','To','From','Botnet','Normal','Background','']
  labelWithoutDigit = re.sub(r'\d', '', label)
  listOfWord = labelWithoutDigit.replace('=', '-').split("-")
  validateArray = []
  if label == 'flow=Background':
    return 'Background'
  else:
    for char in listOfWord:
      if char in extractWordCollection:
        continue
      else:
        validateArray.append(char)
    return '-'.join(validateArray)

def normalization(df):
  ctx= '<PRE-PROCESSING> Normalization'
  start = watcherStart(ctx)

  df['Dur'] = ((df['Dur'] - df['Dur'].min()) / (df['Dur'].max() - df['Dur'].min())* 1000000).astype(int)
  df['SrcAddr'] = ((df['SrcAddr'] - df['SrcAddr'].min()) / (df['SrcAddr'].max() - df['SrcAddr'].min())* 1000000).astype(int)  
  df['DstAddr'] = ((df['DstAddr'] - df['DstAddr'].min()) / (df['DstAddr'].max() - df['DstAddr'].min())* 1000000).astype(int)

  watcherEnd(ctx, start)
  return df

#how to use
def transformation(df, ipv4ToInteger=False, oneHotEncode=False):
  ctx= '<PRE-PROCESSING> Transformation'
  start = watcherStart(ctx)
  #make new label for bot prediciton(1/0)
  df['ActivityLabel'] = df['Label'].str.contains('botnet', case=False, regex=True).astype(int)
  #transform with dictionary
  df['State']= df['State'].map(stateDict).fillna(0.0).astype(int)
  #transform ip to integer
  df.dropna(subset = ["SrcAddr"], inplace=True)
  df.dropna(subset = ["DstAddr"], inplace=True)

  df['Sport'] = pd.factorize(df.Sport)[0]
  df['Dport'] = pd.factorize(df.Dport)[0]

  if(ipv4ToInteger==True):
    df['SrcAddr'] = df['SrcAddr'].apply(ipToInteger).fillna(0)
    df['DstAddr'] = df['DstAddr'].apply(ipToInteger).fillna(0)

  if(oneHotEncode==True): #transform with  one-hot-encode
    categorical_cols = ['Proto','Dir']
    for col in categorical_cols:
      dummy_cols = pd.get_dummies(df[col], drop_first=True, prefix=col)
      df = pd.concat([df,dummy_cols],axis=1)
      df.drop(columns=col, axis=1, inplace=True)
  else:  #transform with dictionary
    df['Proto']= df['Proto'].map(protoDict).fillna(0.0).astype(int)

  watcherEnd(ctx, start)
  return df
