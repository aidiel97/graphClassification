import pandas as pd

from helpers.utilities.watcher import *
from helpers.common.main import *
from pkg.graph.models import *
from pkg.graph.generator import *
from helpers.utilities.dirManagement import checkDir

def extractGraph(df, datasetDetail):
    ctx = 'Graph based analysis - Graph to Tabular' 
    start = watcherStart(ctx)
    srcId = ['Src-Id']
    dstId = ['Dst-Id']
    
    if isinstance(datasetDetail, str):
        datasetName = datasetDetail.replace(OUT_DIR+"split/", "")
        datasetName = datasetName.replace(".csv", "")
        detailData = "test/"
        if "train/" in datasetName:
            datasetName = datasetName.replace("train/","")
            detailData = "train/"
        else:
            datasetName = datasetName.replace("test/","")

        componentDatasetName = datasetName.split('-')
        stringDatasetName = componentDatasetName[0]
        stringSubDatasetName = componentDatasetName[1]
    else:
        stringDatasetName = datasetDetail['stringDatasetName']
        stringSubDatasetName = datasetDetail['selected']
    
    listBotnetAddress = detailBotCount[stringDatasetName][stringSubDatasetName]
    
    #out degree
    node_src_df = df.groupby(srcId).agg(OutDegree = ("Dst-Id", "nunique"),IntensityOutDegree = ("Dst-Id", "count"))
    #group by sourcebytes
    result_src_df = df.groupby(srcId)['SrcBytes'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    result_src_df = result_src_df.rename(columns={'mean': 'MeanSrcBytes','std': 'StdSrcBytes','median': 'MedianSrcBytes','sum': 'SumSrcBytes'})
    #group by dur
    dur_src_df = df.groupby(srcId)['Dur'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    dur_src_df = dur_src_df.rename(columns={'mean': 'MeanDur','std': 'StdDur','median': 'MedianDur','sum': 'SumDur'})
    #group by diff
    diff_src_df = df.groupby(srcId)['Diff'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    diff_src_df = diff_src_df.rename(columns={'mean': 'MeanDiff','std': 'StdDiff','median': 'MedianDiff','sum': 'SumDiff'})
    #group by unix timestamp
    unix_src_df = df.groupby(srcId)['Unix'].agg(['min', 'max']).reset_index()
    unix_src_df = unix_src_df.rename(columns={'min': 'OutStartTime','max': 'OutEndTime'})

    src_df = pd.merge(
        node_src_df, result_src_df, on='Src-Id', how='inner').merge(
            dur_src_df, on='Src-Id', how='inner').merge(
                diff_src_df, on='Src-Id', how='inner').merge(
                    unix_src_df, on='Src-Id', how='inner')
    
    src_df.fillna(0)
    src_df['Address'] = src_df['Src-Id'].str.split('-').str[0]
    src_df['Label'] = src_df['Address'].apply(lambda x: 'botnet' if x in listBotnetAddress else 'normal')
    src_df['SequenceId'] = src_df['Src-Id']
    src_df = src_df[['Address'] + [col for col in src_df.columns if col != 'Address']]

    #in degree
    node_dst_df = df.groupby(dstId).agg(InDegree = ("Src-Id", "nunique"),IntensityInDegree = ("Src-Id", "count"))
    #group by sourcebytes
    result_dst_df = df.groupby(dstId)['SrcBytes'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    result_dst_df = result_dst_df.rename(columns={'mean': 'MeanSrcBytes','std': 'StdSrcBytes','median': 'MedianSrcBytes','sum': 'SumSrcBytes'})
    #group by dur
    dur_dst_df = df.groupby(dstId)['Dur'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    dur_dst_df = dur_dst_df.rename(columns={'mean': 'MeanDur','std': 'StdDur','median': 'MedianDur','sum': 'SumDur'})
    #group by diff
    diff_dst_df = df.groupby(dstId)['Diff'].agg(['mean', 'std', 'median', 'sum']).reset_index()
    diff_dst_df = diff_dst_df.rename(columns={'mean': 'MeanDiff','std': 'StdDiff','median': 'MedianDiff','sum': 'SumDiff'})
    #group by unix timestamp
    unix_dst_df = df.groupby(dstId)['Unix'].agg(['min', 'max']).reset_index()
    unix_dst_df = unix_dst_df.rename(columns={'min': 'InStartTime','max': 'InEndTime'})

    dst_df = pd.merge(
        node_dst_df, result_dst_df, on='Dst-Id', how='inner').merge(
            dur_dst_df, on='Dst-Id', how='inner').merge(
                diff_dst_df, on='Dst-Id', how='inner').merge(
                    unix_dst_df, on='Dst-Id', how='inner')
    
    dst_df.fillna(0)
    dst_df['Address'] = dst_df['Dst-Id'].str.split('-').str[0]
    dst_df['Label'] = dst_df['Address'].apply(lambda x: 'botnet' if x in listBotnetAddress else 'normal')
    dst_df['SequenceId'] = dst_df['Dst-Id']
    dst_df = dst_df[['Address'] + [col for col in dst_df.columns if col != 'Address']]
    
    checkDir(OUT_DIR+'extract/')
    checkDir(OUT_DIR+'extract/'+detailData)
    src_df.to_csv(OUT_DIR+'extract/'+detailData+stringDatasetName+'-'+stringSubDatasetName+'-out.csv', index=False)
    dst_df.to_csv(OUT_DIR+'extract/'+detailData+stringDatasetName+'-'+stringSubDatasetName+'-in.csv', index=False)
    # Print a completion message
    print("Processing complete.")
    watcherEnd(ctx, start)
