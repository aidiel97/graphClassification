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
    ('Generate Graph Dataset (from specific dataset)', grp.singleData),
    ('Generate Graph Dataset (from all dataset)', grp.executeAllData),
    ('Generate Machine Learning Models', ml.trainingAllAlgorithm),
    ('Classified Network Graph', ml.executeAllDataGraph),
  ]
  cli.menu(listMenu)