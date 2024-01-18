import time
import csv

from helpers.utilities.dirManagement import checkDir
from datetime import datetime

def watcherStart(processName):
  start = time.time()
  print('\n===========================================| [ START ] '+processName)

  return start

def watcherEnd(processName, start=time.time(), record=False):
  end = time.time()
  processingTime = '{:.3f}'.format(end - start)
  
  if record:
    field_names = ['ProcessName','StartAt','EndAt','ProcessingTime']
    dict = {
      "ProcessName": processName,
      "StartAt":datetime.fromtimestamp(start),
      "EndAt":datetime.fromtimestamp(end),
      "ProcessingTime": end-start,
    }

    checkDir(OUT_DIR+'')
    with open(OUT_DIR+'ActivityLog.csv', 'a', newline='') as csv_file:
      dict_object = csv.DictWriter(csv_file, fieldnames=field_names) 
      dict_object.writerow(dict)
  
  print('\n===========================================| [  END  ] '+processName+' ('+str(processingTime)+' s)')
