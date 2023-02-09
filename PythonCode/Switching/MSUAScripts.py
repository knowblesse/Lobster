import numpy as np
import matplotlib.pyplot as plt
from Switching.SwitchingHelper import parseAllData, mahal
plt.rcParams["font.family"] = "Noto Sans"

tankName =  '#20JUN1-200916-111545_PL' # Good
#tankName = '#21JAN5-210622-180202_PL' # wandering
#tankName = '#21JAN2-210406-190737_IL'

data = parseAllData(tankName)
neural_data = data['neural_data']
ParsedData = data['ParsedData']
zoneClass = data['zoneClass']
numTrial = data['numTrial']
midPointTimes = data['midPointTimes']
Trials = data['Trials']
IRs = data['IRs']
AEResult = data['AEResult']


withinNesting = mahal(neural_data[zoneClass==0, :], neural_data[zoneClass==0, :])
withinForaging = mahal(neural_data[zoneClass==1, :], neural_data[zoneClass==1, :])
withinEncounter = mahal(neural_data[zoneClass==2, :], neural_data[zoneClass==2, :])

print(f'N:{np.mean(withinNesting)}, F:{np.mean(withinForaging)}, E:{np.mean(withinEncounter)}')


