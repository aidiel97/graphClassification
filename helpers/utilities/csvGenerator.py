import csv
from datetime import datetime
from helpers.utilities.watcher import *
from helpers.common.globalConfig import OUT_DIR

def similarityResult(botnetUniqueSelection, normalUniqueSelection, botnetRecord, normalRecord, samplingRule, decompositor, similarityResult):
  ctx= 'Export Similarity Result'
  start= watcherStart(ctx)
  # list of column names 
  field_names = ['CreatedAt', 'botnetUniqueSelection','normalUniqueSelection','botnetRecord','normalRecord','samplingRule', 'decompositor', 'similarityResult']
  # Dictionary
  dict = {
    "CreatedAt": datetime.now(),
    "botnetUniqueSelection": botnetUniqueSelection,
    "normalUniqueSelection": normalUniqueSelection,
    "botnetRecord": botnetRecord,
    "normalRecord": normalRecord,
    "samplingRule": samplingRule,
    "decompositor": decompositor,
    "similarityResult": similarityResult
  }

  with open('data/similarity_results.csv', 'a', newline='') as csv_file:
    dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
    dict_object.writerow(dict)

  watcherEnd(ctx, start)

def classificationResult(classificationContext, algorithm, tn, fp, fn, tp):
  ctx= 'Export Classification Result'
  start= watcherStart(ctx)
  # list of column names 
  field_names = ['CreatedAt', 'Algorithm', 'TN', 'FP', 'FN', 'TP', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'ClassificationContext']
  accuracy = (tp+tn)/(tp+tn+fp+fn)
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  f1 = (2*precision*recall)/(precision+recall)
  # Dictionary
  dict = {
    "CreatedAt": datetime.now(),
    "Algorithm":algorithm,
    "ClassificationContext":classificationContext,
    "TN": tn,
    "FP": fp,
    "FN": fn,
    "TP": tp,
    "Accuracy": accuracy,
    "Precision": precision,
    "Recall": recall,
    "F1-score": f1,
  }

  with open(OUT_DIR+'classification_results.csv', 'a', newline='') as csv_file:
    dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
    dict_object.writerow(dict)

  watcherEnd(ctx, start)

def exportWithObject(object, fileName):
  ctx= 'Export data to: '+fileName
  start= watcherStart(ctx)
  # list of column names 
  field_names = object.keys()

  with open(fileName, 'a', newline='') as csv_file:
    dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
    dict_object.writerow(object)

  watcherEnd(ctx, start)

def exportWithArrayOfObject(arrayObject, fileName):
  ctx= 'Export data to: '+fileName
  start= watcherStart(ctx)
  
  field_names = arrayObject[0].keys()
  with open(fileName, 'a', newline='') as csv_file:
    obj = csv.DictWriter(csv_file, fieldnames=field_names)
    obj.writeheader()

  for object in arrayObject:
    # list of column names 
    field_names = object.keys()

    with open(fileName, 'a', newline='') as csv_file:
      dict_object = csv.DictWriter(csv_file, fieldnames=field_names)
      dict_object.writerow(object)

  watcherEnd(ctx, start)