import helpers.utilities.dataLoader as loader
import pkg.preProcessing.transform as pp

from helpers.utilities.watcher import *
from helpers.common.main import *
from pkg.graph.models import *
from pkg.graph.generator import *
from pkg.graph.extractor import *

def dftoGraph(datasetDetail):
    ctx = 'Graph based analysis - DF to Graph'
    start = watcherStart(ctx)

    # check the variable is string or dictionary
    if isinstance(datasetDetail, str):
        raw_df = loader.rawCsv(datasetDetail)
    else:
        raw_df = loader.binetflow(
            datasetDetail['datasetName'],
            datasetDetail['selected'],
            datasetDetail['stringDatasetName'])
    
    raw_df['Unix'] = raw_df['StartTime'].apply(pp.timeToUnix).fillna(0)
    
    # Function to calculate the "Diff" column
    def calculate_diff(row):
        index = row.name
        if index > 0 and raw_df.loc[index, 'SrcAddr'] == raw_df.loc[index - 1, 'SrcAddr']:
            return row['Unix'] - raw_df.loc[index - 1, 'Unix']
        return None
    
    raw_df['Diff'] = raw_df.apply(calculate_diff, axis=1)

    # Initialize variables
    x = 0
    prev_src_addr = None
    # Custom function to calculate "Src-Id"
    def calculate_src_id(row):
        nonlocal x, prev_src_addr
        if prev_src_addr is None or row['SrcAddr'] != prev_src_addr:
            x = 0
        elif row['Diff'] > DEFAULT_TIME_GAP:
            x += 1
        prev_src_addr = row['SrcAddr']
        return row['SrcAddr'] + "-" + str(x)

    # Custom function to calculate "Src-Id"
    prev_dst_addr = None
    prev_srcId_addr = None
    def calculate_dst_id(row):
        nonlocal x, prev_dst_addr, prev_srcId_addr
        if (prev_dst_addr is None and prev_dst_addr is None) or row['DstAddr'] != prev_dst_addr:
            x = 0
        elif row['Src-Id'] != prev_srcId_addr:
            x += 1
        prev_dst_addr = row['DstAddr']
        prev_srcId_addr = row['Src-Id']
        return row['DstAddr'] + "-" + str(x)

    raw_df = raw_df.sort_values(by=['SrcAddr', 'Unix'])
    raw_df = raw_df.reset_index(drop=True)
    raw_df['Src-Id'] = raw_df.apply(calculate_src_id, axis=1)

    raw_df = raw_df.sort_values(by=['DstAddr', 'Src-Id'])
    raw_df = raw_df.reset_index(drop=True)
    raw_df['Dst-Id'] = raw_df.apply(calculate_dst_id, axis=1)

    raw_df['Diff'] = raw_df['Diff'].fillna(0)
    raw_df = raw_df.fillna('-')

    extractGraph(raw_df, datasetDetail)

    watcherEnd(ctx, start, True)
