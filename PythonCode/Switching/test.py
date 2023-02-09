import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Switching.SwitchingHelper import parseAllData, getZoneLDA

FolderPath = Path(r'D:/Data/Lobster/FineDistanceDataset')
outputData = []

for tank in [p for p in FolderPath.glob('#*')]:
    array2append = []
    tankName = re.search('#.*', str(tank))[0]

    data = parseAllData(tankName)
    zoneClass = data['zoneClass']
    numTrial = data['numTrial']
    midPointTimes = data['midPointTimes']
    Trials = data['Trials']
    AEResult = data['AEResult']
    ParsedData = data['ParsedData']

    ratioWanderReady = [0,0]
    for trial in np.arange(1, numTrial):
        latency2HeadEntry = ParsedData[trial, 1][0, 0] # first IRON from TRON

        # get behavior types
        if latency2HeadEntry >= 6:
            ratioWanderReady[0] += 1
        else:
            ratioWanderReady[1] += 1
    print(f'{tankName} : {ratioWanderReady[0] / sum(ratioWanderReady)}')