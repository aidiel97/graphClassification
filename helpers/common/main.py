from helpers.common.globalConfig import DATASET_LOCATION, CTU_DIR, NCC_DIR, NCC2_DIR, NCC_GRAPH_DIR, DEFAULT_MACHINE_LEARNING_TRAIN_PROPORTION, DEFAULT_TIME_GAP

defaultTimeGap = DEFAULT_TIME_GAP
defaultTrainProportion = DEFAULT_MACHINE_LEARNING_TRAIN_PROPORTION
datasetLocation = DATASET_LOCATION
ctuLoc = CTU_DIR
nccLoc = NCC_DIR
ncc2Loc = NCC2_DIR
nccGraphLoc = NCC_GRAPH_DIR

ctuPcap = {
  'scenario1': datasetLocation+ctuLoc+'/1/botnet-capture-20110810-neris.pcap',
  'scenario2': datasetLocation+ctuLoc+'/2/capture20110811.binetflow',
  'scenario3': datasetLocation+ctuLoc+'/3/capture20110812.binetflow',
  'scenario4': datasetLocation+ctuLoc+'/4/capture20110815.binetflow',
  'scenario5': datasetLocation+ctuLoc+'/5/capture20110815-2.binetflow',
  'scenario6': datasetLocation+ctuLoc+'/6/capture20110816.binetflow',
  'scenario7': datasetLocation+ctuLoc+'/7/capture20110816-2.binetflow',
  'scenario8': datasetLocation+ctuLoc+'/8/capture20110816-3.binetflow',
  'scenario9': datasetLocation+ctuLoc+'/9/capture20110817.binetflow',
  'scenario10': datasetLocation+ctuLoc+'/10/capture20110818.binetflow',
  'scenario11': datasetLocation+ctuLoc+'/11/capture20110818-2.binetflow',
  'scenario12': datasetLocation+ctuLoc+'/12/capture20110819.binetflow',
  'scenario13': datasetLocation+ctuLoc+'/13/capture20110815-3.binetflow',
}

#all CTU-13 dataset scenarios
ctuOnline = {
  'scenario1': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-42/detailed-bidirectional-flow-labels/capture20110810.binetflow',
  'scenario2': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-43/detailed-bidirectional-flow-labels/capture20110811.binetflow',
  'scenario3': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-44/detailed-bidirectional-flow-labels/capture20110812.binetflow',
  'scenario4': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-45/detailed-bidirectional-flow-labels/capture20110815.binetflow',
  'scenario5': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-46/detailed-bidirectional-flow-labels/capture20110815-2.binetflow',
  'scenario6': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-47/detailed-bidirectional-flow-labels/capture20110816.binetflow',
  'scenario7': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-48/detailed-bidirectional-flow-labels/capture20110816-2.binetflow',
  'scenario8': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-49/detailed-bidirectional-flow-labels/capture20110816-3.binetflow',
  'scenario9': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-50/detailed-bidirectional-flow-labels/capture20110817.binetflow',
  'scenario10': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-51/detailed-bidirectional-flow-labels/capture20110818.binetflow',
  'scenario11': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-52/detailed-bidirectional-flow-labels/capture20110818-2.binetflow',
  'scenario12': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-53/detailed-bidirectional-flow-labels/capture20110819.binetflow',
  'scenario13': 'https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-54/detailed-bidirectional-flow-labels/capture20110815-3.binetflow',
}

ctu = {
  'scenario1': datasetLocation+ctuLoc+'/1/capture20110810.binetflow',
  'scenario2': datasetLocation+ctuLoc+'/2/capture20110811.binetflow',
  'scenario3': datasetLocation+ctuLoc+'/3/capture20110812.binetflow',
  'scenario4': datasetLocation+ctuLoc+'/4/capture20110815.binetflow',
  'scenario5': datasetLocation+ctuLoc+'/5/capture20110815-2.binetflow',
  'scenario6': datasetLocation+ctuLoc+'/6/capture20110816.binetflow',
  'scenario7': datasetLocation+ctuLoc+'/7/capture20110816-2.binetflow',
  'scenario8': datasetLocation+ctuLoc+'/8/capture20110816-3.binetflow',
  'scenario9': datasetLocation+ctuLoc+'/9/capture20110817.binetflow',
  'scenario10': datasetLocation+ctuLoc+'/10/capture20110818.binetflow',
  'scenario11': datasetLocation+ctuLoc+'/11/capture20110818-2.binetflow',
  'scenario12': datasetLocation+ctuLoc+'/12/capture20110819.binetflow',
  'scenario13': datasetLocation+ctuLoc+'/13/capture20110815-3.binetflow',
}

ncc = {
  'scenario1': datasetLocation+nccLoc+'/scenario_dataset_1/dataset_result.binetflow',
  'scenario2': datasetLocation+nccLoc+'/scenario_dataset_2/dataset_result.binetflow',
  'scenario3': datasetLocation+nccLoc+'/scenario_dataset_3/dataset_result.binetflow',
  'scenario4': datasetLocation+nccLoc+'/scenario_dataset_4/dataset_result.binetflow',
  'scenario5': datasetLocation+nccLoc+'/scenario_dataset_5/dataset_result.binetflow',
  'scenario6': datasetLocation+nccLoc+'/scenario_dataset_6/dataset_result.binetflow',
  'scenario7': datasetLocation+nccLoc+'/scenario_dataset_7/dataset_result.binetflow',
  'scenario8': datasetLocation+nccLoc+'/scenario_dataset_8/dataset_result.binetflow',
  'scenario9': datasetLocation+nccLoc+'/scenario_dataset_9/dataset_result.binetflow',
  'scenario10': datasetLocation+nccLoc+'/scenario_dataset_10/dataset_result.binetflow',
  'scenario11': datasetLocation+nccLoc+'/scenario_dataset_11/dataset_result.binetflow',
  'scenario12': datasetLocation+nccLoc+'/scenario_dataset_12/dataset_result.binetflow',
  'scenario13': datasetLocation+nccLoc+'/scenario_dataset_13/dataset_result.binetflow',
}

ncc2 = {
  'scenario1': datasetLocation+ncc2Loc+'/sensor1/sensor1.binetflow',
  'scenario2': datasetLocation+ncc2Loc+'/sensor2/sensor2.binetflow',
  'scenario3': datasetLocation+ncc2Loc+'/sensor3/sensor3.binetflow',
}

ncc2AllScenarios = {
  'scenario1': datasetLocation+ncc2Loc+'/all-sensors/sensors-all.binetflow',
}

nccGraphCTU = {
    'scenario1' : datasetLocation+nccGraphLoc+'/ctu-scenario1-in.csv',
    'scenario2' : datasetLocation+nccGraphLoc+'/ctu-scenario2-in.csv',
    'scenario3' : datasetLocation+nccGraphLoc+'/ctu-scenario3-in.csv',
    'scenario4' : datasetLocation+nccGraphLoc+'/ctu-scenario4-in.csv',
    'scenario5' : datasetLocation+nccGraphLoc+'/ctu-scenario5-in.csv',
    'scenario6' : datasetLocation+nccGraphLoc+'/ctu-scenario6-in.csv',
    'scenario7' : datasetLocation+nccGraphLoc+'/ctu-scenario7-in.csv',
    'scenario8' : datasetLocation+nccGraphLoc+'/ctu-scenario8-in.csv',
    'scenario9' : datasetLocation+nccGraphLoc+'/ctu-scenario9-in.csv',
    'scenario10' : datasetLocation+nccGraphLoc+'/ctu-scenario10-in.csv',
    'scenario11' : datasetLocation+nccGraphLoc+'/ctu-scenario11-in.csv',
    'scenario12' : datasetLocation+nccGraphLoc+'/ctu-scenario12-in.csv',
    'scenario13' : datasetLocation+nccGraphLoc+'/ctu-scenario13-in.csv',
}

nccGraphNCC = {
    'scenario1' : datasetLocation+nccGraphLoc+'/ncc-scenario1-in.csv',
    'scenario2' : datasetLocation+nccGraphLoc+'/ncc-scenario2-in.csv',
    'scenario3' : datasetLocation+nccGraphLoc+'/ncc-scenario3-in.csv',
    'scenario4' : datasetLocation+nccGraphLoc+'/ncc-scenario4-in.csv',
    'scenario5' : datasetLocation+nccGraphLoc+'/ncc-scenario5-in.csv',
    'scenario6' : datasetLocation+nccGraphLoc+'/ncc-scenario6-in.csv',
    'scenario7' : datasetLocation+nccGraphLoc+'/ncc-scenario7-in.csv',
    'scenario8' : datasetLocation+nccGraphLoc+'/ncc-scenario8-in.csv',
    'scenario9' : datasetLocation+nccGraphLoc+'/ncc-scenario9-in.csv',
    'scenario10' : datasetLocation+nccGraphLoc+'/ncc-scenario10-in.csv',
    'scenario11' : datasetLocation+nccGraphLoc+'/ncc-scenario11-in.csv',
    'scenario12' : datasetLocation+nccGraphLoc+'/ncc-scenario12-in.csv',
    'scenario13' : datasetLocation+nccGraphLoc+'/ncc-scenario13-in.csv',
}

nccGraphNCC2 = {
    'scenario1' : datasetLocation+nccGraphLoc+'/ncc2-scenario1-in.csv',
    'scenario2' : datasetLocation+nccGraphLoc+'/ncc2-scenario2-in.csv',
    'scenario3' : datasetLocation+nccGraphLoc+'/ncc2-scenario3-in.csv',
}

listAvailableDatasets=[
  {
    'name':'NCC (Periodic Botnet)',
    'shortName': 'ncc',
    'list': ncc
  },
  {
    'name':'NCC-2 (Simultaneous Botnet)',
    'shortName': 'ncc2',
    'list': ncc2
  },
  {
    'name':'CTU-13 (Local Source)',
    'shortName': 'ctu',
    'list': ctu
  },
  {
    'name':'CTU-13 (Online Source)',
    'shortName': 'ctu',
    'list': ctuOnline
  },
]

listAvailableGraphDatasets=[
  {
    'name':'Graph-NCC',
    'shortName': 'GraphNcc',
    'list': nccGraphNCC
  },
  {
    'name':'Graph-NCC-2',
    'shortName': 'GraphNcc2',
    'list': nccGraphNCC2
  },
  {
    'name':'Graph-CTU-13',
    'shortName': 'GraphCtu',
    'list': nccGraphCTU
  },
]

#state dictionary (source: Nagabhushan S Baddi)
stateDict = {'': 1, 'FSR_SA': 30, '_FSA': 296, 'FSRPA_FSA': 77, 'SPA_SA': 31, 'FSA_SRA': 1181, 'FPA_R': 46, 'SPAC_SPA': 37, 'FPAC_FPA': 2, '_R': 1, 'FPA_FPA': 784, 'FPA_FA': 66, '_FSRPA': 1, 'URFIL': 431, 'FRPA_PA': 5, '_RA': 2, 'SA_A': 2, 'SA_RA': 125, 'FA_FPA': 17, 'FA_RA': 14, 'PA_FPA': 48, 'URHPRO': 380, 'FSRPA_SRA': 8, 'R_':541, 'DCE': 5, 'SA_R': 1674, 'SA_': 4295, 'RPA_FSPA': 4, 'FA_A': 17, 'FSPA_FSPAC': 7, 'RA_': 2230, 'FSRPA_SA': 255, 'NNS': 47, 'SRPA_FSPAC': 1, 'RPA_FPA': 42, 'FRA_R': 10, 'FSPAC_FSPA': 86, 'RPA_R': 3, '_FPA': 5, 'SREC_SA': 1, 'URN': 339, 'URO': 6, 'URH': 3593, 'MRQ': 4, 'SR_FSA': 1, 'SPA_SRPAC': 1, 'URP': 23598, 'RPA_A': 1, 'FRA_': 351, 'FSPA_SRA': 91, 'FSA_FSA': 26138, 'PA_': 149, 'FSRA_FSPA': 798, 'FSPAC_FSA': 11, 'SRPA_SRPA': 176, 'SA_SA': 33, 'FSPAC_SPA': 1, 'SRA_RA': 78, 'RPAC_PA': 1, 'FRPA_R': 1, 'SPA_SPA': 2989, 'PA_RA': 3, 'SPA_SRPA': 4185, 'RA_FA': 8, 'FSPAC_SRPA': 1, 'SPA_FSA': 1, 'FPA_FSRPA': 3, 'SRPA_FSA': 379, 'FPA_FRA': 7, 'S_SRA': 81, 'FSA_SA': 6, 'State': 1, 'SRA_SRA': 38, 'S_FA': 2, 'FSRPAC_SPA': 7, 'SRPA_FSPA': 35460, 'FPA_A': 1, 'FSA_FPA': 3, 'FRPA_RA': 1, 'FSAU_SA': 1, 'FSPA_FSRPA': 10560, 'SA_FSA': 358, 'FA_FRA': 8, 'FSRPA_SPA': 2807, 'FSRPA_FSRA': 32, 'FRA_FPA': 6, 'FSRA_FSRA': 3, 'SPAC_FSRPA': 1, 'FS_': 40, 'FSPA_FSRA': 798, 'FSAU_FSA': 13, 'A_R': 36, 'FSRPAE_FSPA': 1, 'SA_FSRA': 4, 'PA_PAC': 3, 'FSA_FSRA': 279, 'A_A': 68, 'REQ': 892, 'FA_R': 124, 'FSRPA_SRPA': 97, 'FSPAC_FSRA':20, 'FRPA_RPA': 7, 'FSRA_SPA': 8, 'INT': 85813, 'FRPA_FRPA': 6, 'SRPAC_FSPA': 4, 'SPA_SRA': 808, 'SA_SRPA': 1, 'SPA_FSPA': 2118, 'FSRAU_FSA': 2, 'RPA_PA': 171,'_SPA': 268, 'A_PA': 47, 'SPA_FSRA': 416, 'FSPA_FSRPAC': 2, 'PAC_PA': 5, 'SRPA_SPA': 9646, 'SRPA_FSRA': 13, 'FPA_FRPA': 49, 'SRA_SPA': 10, 'SA_SRA': 838, 'PA_PA': 5979, 'FPA_RPA': 27, 'SR_RA': 10, 'RED': 4579, 'CON': 2190507, 'FSRPA_FSPA':13547, 'FSPA_FPA': 4, 'FAU_R': 2, 'ECO': 2877, 'FRPA_FPA': 72, 'FSAU_SRA': 1, 'FRA_FA': 8, 'FSPA_FSPA': 216341, 'SEC_RA': 19, 'ECR': 3316, 'SPAC_FSPA': 12, 'SR_A': 34, 'SEC_': 5, 'FSAU_FSRA': 3, 'FSRA_FSRPA': 11, 'SRC': 13, 'A_RPA': 1, 'FRA_PA': 3, 'A_RPE': 1, 'RPA_FRPA': 20, '_SRA': 74, 'SRA_FSPA': 293, 'FPA_': 118, 'FSRPAC_FSRPA': 2, '_FA': 1, 'DNP': 1, 'FSRPA_FSRPA': 379, 'FSRA_SRA': 14, '_FRPA': 1, 'SR_': 59, 'FSPA_SPA': 517, 'FRPA_FSPA': 1, 'PA_A': 159, 'PA_SRA': 1, 'FPA_RA': 5, 'S_': 68710, 'SA_FSRPA': 4, 'FSA_FSRPA': 1, 'SA_SPA': 4, 'RA_A': 5, '_SRPA': 9, 'S_FRA': 156, 'FA_FRPA': 1, 'PA_R': 72, 'FSRPAEC_FSPA': 1, '_PA': 7, 'RA_S': 1, 'SA_FR': 2, 'RA_FPA': 6, 'RPA_': 5, '_FSPA': 2395, 'FSA_FSPA': 230, 'UNK': 2, 'A_RA': 9, 'FRPA_': 6, 'URF': 10, 'FS_SA': 97, 'SPAC_SRPA': 8, 'S_RPA': 32, 'SRPA_SRA': 69, 'SA_RPA': 30, 'PA_FRA': 4, 'FSRA_SA': 49, 'FSRA_FSA': 206, 'PAC_RPA': 1, 'SRA_': 18, 'FA_': 451, 'S_SA': 6917, 'FSPA_SRPA': 427, 'TXD': 542,'SRA_SA': 1514, 'FSPA_FA': 1, 'FPA_FSPA': 10, 'RA_PA': 3, 'SRA_FSA': 709, 'SRPA_SPAC': 3, 'FSPAC_FSRPA': 10, 'A_': 191, 'URNPRO': 2, 'PA_RPA': 81, 'FSPAC_SRA':1, 'SRPA_FSRPA': 3054, 'SPA_': 1, 'FA_FA': 259, 'FSPA_SA': 75, 'SR_SRA': 1, 'FSA_': 2, 'SRPA_SA': 406, 'SR_SA': 3119, 'FRPA_FA': 1, 'PA_FRPA': 13, 'S_R': 34, 'FSPAEC_FSPAE': 3, 'S_RA': 61105, 'FSPA_FSA': 5326, '_SA': 20, 'SA_FSPA': 15, 'SRPAC_SPA': 8, 'FPA_PA': 19, 'FSRPAE_FSA': 1, 'S_A': 1, 'RPA_RPA': 3, 'NRS': 6, 'RSP': 115, 'SPA_FSRPA': 1144, 'FSRPAC_FSPA': 139}
#dicts to convert protocols and state to integers
protoDict = {'arp': 5, 'unas': 13, 'udp': 1, 'rtcp': 7, 'pim': 3, 'udt': 11, 'esp': 12, 'tcp' : 0, 'rarp': 14, 'ipv6-icmp': 9, 'rtp': 2, 'ipv6': 10, 'ipx/spx': 6, 'icmp': 4, 'igmp' : 8}

#listBotIPinDataset
botIP = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205', '147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']

botCount13Scenario = {
  'scenario1' : [botIP[0]],
  'scenario2' : [botIP[0]],
  'scenario3' : [botIP[0]],
  'scenario4' : [botIP[0]],
  'scenario5' : [botIP[0]],
  'scenario6' : [botIP[0]],
  'scenario7' : [botIP[0]],
  'scenario8' : [botIP[0]],
  'scenario9' : botIP,
  'scenario10' : botIP,
  'scenario11' : botIP[:3],
  'scenario12' : botIP[:3],
  'scenario13' : [botIP[0]],
}

botCount3Scenario = {
  'scenario1' : botIP,
  'scenario2' : botIP,
  'scenario3' : botIP,
}

detailBotCount = {
  'ctu': botCount13Scenario,
  'ncc': botCount13Scenario,
  'ncc2': botCount3Scenario
}