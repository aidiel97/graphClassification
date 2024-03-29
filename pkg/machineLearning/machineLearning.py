from helpers.utilities.watcher import *
import helpers.utilities.csvGenerator as csv

import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


algorithmDict = {
  'decisionTree': DecisionTreeClassifier(),
  'randomForest': RandomForestClassifier(),
  'naiveBayes': GaussianNB(),
  'logisticRegression' : LogisticRegression(),
  'xGBoost': GradientBoostingClassifier(),
  # 'svc' : SVC(),
  # 'knn': KNeighborsClassifier(),
  # 'ann': MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(15,), random_state=1)
}

def modelFileName(algorithm): return 'collections/'+algorithm+'.pkl'

def modelling(x, y, algorithm='randomForest'):
  ctx= 'Training Model With: '+algorithm.upper()
  start= watcherStart(ctx)

  model = algorithmDict[algorithm]
  model.fit(x, y)
  pickle.dump(model, open(modelFileName(algorithm), 'wb'))

  watcherEnd(ctx, start)

def classification(x, algorithm='randomForest'):
  ctx= 'Classifying Data - '+algorithm
  start= watcherStart(ctx)

  model = pickle.load(open(modelFileName(algorithm), 'rb'))
  predictionResult = model.predict(x)

  watcherEnd(ctx, start)
  return predictionResult

def evaluation(ctx, y, predictionResult, algorithm='randomForest'):
  tn, fp, fn, tp = confusion_matrix(y, predictionResult).ravel()

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