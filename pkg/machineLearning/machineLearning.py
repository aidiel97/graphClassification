from helpers.utilities.watcher import *
import helpers.utilities.csvGenerator as csv

import time
import pickle
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from helpers.common.globalConfig import OUT_DIR


algorithmDict = {
  'decisionTree': DecisionTreeClassifier(criterion='entropy', max_depth=13),
  'logisticRegression' : LogisticRegression(),
  'naiveBayes': MultinomialNB(alpha=0.5, fit_prior=False),

  'adaboost': AdaBoostClassifier(n_estimators=600, learning_rate=1.0),
  'extraTree': ExtraTreesClassifier(n_estimators=400, criterion='entropy'),
  'xGBoost': GradientBoostingClassifier(),
  'randomForest': RandomForestClassifier(n_estimators=1000, criterion='entropy', max_features='log2'),
  
  'knn': KNeighborsClassifier(n_neighbors=13, metric='manhattan', weights='uniform'),
  
  # 'svc' : SVC(),
  # 'ann': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
}

def modelFileName(algorithm): return OUT_DIR+''+algorithm+'.pkl'

def modelling(x, y, graphDetail, algorithm='randomForest'):
  ctx= str(time.time())+' | Training Model With: '+algorithm.upper()
  start= watcherStart(ctx)

  model = algorithmDict[algorithm]
  model.fit(x, y)
  pickle.dump(model, open(modelFileName(algorithm+'-'+graphDetail), 'wb'))

  watcherEnd(ctx, start)

def classification(x, graphDetail, algorithm='randomForest'):
  ctx= 'Classifying Data - '+algorithm
  start= watcherStart(ctx)

  model = pickle.load(open(modelFileName(algorithm+'-'+graphDetail), 'rb'))
  predictionResult = model.predict(x)

  watcherEnd(ctx, start)
  return predictionResult

def evaluation(ctx, y, predictionResult, algorithm='randomForest'):
  tn, fp, fn, tp = confusion_matrix(y, predictionResult, labels=[0,1]).ravel()

  print('\nAlgorithm\t\t\t: '+algorithm)
  print('\nTotal input data\t\t\t: '+str(y.shape[0]))
  print('TN (predict result 0, actual 0)\t\t: '+str(tn))
  print('FP (predict result 1, actual 0)\t\t: '+str(fp))
  print('FN (predict result 0, actual 1)\t\t: '+str(fn))
  print('TP (predict result 1, actual 1)\t\t: '+str(tp))
  print('Accuracy\t\t\t\t: '+str((tp+tn)/(tp+tn+fp+fn)))
  print('TPR\t\t\t\t\t: '+str((tp)/(tp+fn)))
  print('TNR\t\t\t\t\t: '+str((tn)/(tn+fp)))

  csv.classificationResult(ctx, algorithm, tn, fp, fn, tp)