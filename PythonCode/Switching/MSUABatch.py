import re
from pathlib import Path
import numpy as np
from tqdm import tqdm
from Switching.SwitchingHelper import parseAllData, mahal

FolderPath = Path(r'D:/Data/Lobster/FineDistanceDataset')
outputData = []
pbar = tqdm([p for p in FolderPath.glob('#*')])
for tank in pbar:
    tankName = re.search('#.*', str(tank))[0]

    data = parseAllData(tankName)
    neural_data = data['neural_data']
    ParsedData = data['ParsedData']
    zoneClass = data['zoneClass']
    numTrial = data['numTrial']
    midPointTimes = data['midPointTimes']
    Trials = data['Trials']
    IRs = data['IRs']
    AEResult = data['AEResult']

    withinNesting = mahal(neural_data[zoneClass == 0, :], neural_data[zoneClass == 0, :])
    withinForaging = mahal(neural_data[zoneClass == 1, :], neural_data[zoneClass == 1, :])
    withinEncounter = mahal(neural_data[zoneClass == 2, :], neural_data[zoneClass == 2, :])


    outputData.append([tankName,
                       withinNesting,
                       withinForaging,
                       withinEncounter
                       ])


np.savetxt('C:/Users/Knowblesse/Desktop/stds.csv', np.array(outputData), delimiter=',', fmt='%s')
