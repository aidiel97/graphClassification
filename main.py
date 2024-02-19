"""Paralel Botnet Data Forensic"""
"""Writen By: M. Aidiel Rachman Putra"""
"""Organization: Net-Centic Computing Laboratory | Institut Teknologi Sepuluh Nopember"""

import warnings
warnings.simplefilter(action='ignore')
import pkg.graph.domain as grp
import interfaces.cli.main as cli
import pkg.machineLearning.domain as ml
import pkg.datasetAnalysis.domain as analysis

if __name__ == "__main__":
  listMenu = [
    ('Split dataset and get optimal time gap', analysis.splitAndTimeGap),
    ('Generate graph dataset (from all dataset)', grp.executeAllData),
    ('Training the in-degree graph dataset', ml.trainingInGraph),
    ('Training the out-degree graph dataset', ml.trainingOutGraph),
    ('Classify network graph', ml.executeAllDataGraph),
    ('Combine two prediction result', ml.combinePredictionResult),
  ]
  cli.menu(listMenu)