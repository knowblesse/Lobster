"""
drawLDAGraph
@2022 Knowblesse
2022NOV07
"""

import numpy as np
import matplotlib.pyplot as plt
from Switching.SwitchingHelper import parseAllData, getZoneLDA
plt.rcParams["font.family"] = "Noto Sans"
def drawLDAResult(neural_data_transformed, zoneClass, centroids, tankName, dotNumber=200, drawOnlyCloserObjects=False, useOldFigure=False):
    if useOldFigure:
        fig = plt.gcf()
    else:
        fig = plt.figure(figsize=(10,10))
    fig.clf()
    ax = fig.subplots(1,1)

    # Select only partial data
    if drawOnlyCloserObjects:
        # get the closest object
        zC0 = np.where(zoneClass==0)[0][
            np.argsort(
                np.sum((np.array(centroids['nest']) - neural_data_transformed[zoneClass == 0, :]) ** 2, 1) ** 0.5)[:dotNumber*10]]
        zC1 = np.where(zoneClass == 1)[0][
            np.argsort(
                np.sum((np.array(centroids['foraging']) - neural_data_transformed[zoneClass == 1, :]) ** 2, 1) ** 0.5)[:dotNumber*10]]
        zC2 = np.where(zoneClass == 2)[0][
            np.argsort(
                np.sum((np.array(centroids['encounter']) - neural_data_transformed[zoneClass == 2, :]) ** 2, 1) ** 0.5)[:dotNumber*10]]
        # select random from it
        zC0 = np.random.choice(zC0, dotNumber)
        zC1 = np.random.choice(zC1, dotNumber)
        zC2 = np.random.choice(zC2, dotNumber)
    else:
        zC0 = np.random.choice(np.where(zoneClass==0)[0], dotNumber)
        zC1 = np.random.choice(np.where(zoneClass==1)[0], dotNumber)
        zC2 = np.random.choice(np.where(zoneClass==2)[0], dotNumber)

    ax.scatter(neural_data_transformed[zC0, 0], neural_data_transformed[zC0, 1], color=np.array([85,98,112,90])/255, s=15)
    ax.scatter(neural_data_transformed[zC1, 0], neural_data_transformed[zC1, 1], color=np.array([78,205,196,90])/255, s=15)
    #ax.scatter(neural_data_transformed[zC2, 0], neural_data_transformed[zC2, 1], color=np.array([255,107,107,90])/255, s=15)

    ax.scatter(centroids['nest'][0], centroids['nest'][1], color=np.array([85,98,112])/255, edgecolor='grey', marker='D', s=100)
    ax.scatter(centroids['foraging'][0], centroids['foraging'][1], color=np.array([78,205,196])/255, edgecolor='grey', marker='D', s=100)
    #ax.scatter(centroids['encounter'][0], centroids['encounter'][1], color=np.array([255,107,107])/255, edgecolor='grey', marker='D', s=100)

    ax.legend(["Nesting", "Foraging", "Encounter"])
    ax.set_ylim(-1, 1)
    ax.set_xlim(-0.5, 1)
    ax.set_title(tankName + "- LDA")
    return (fig, ax)
def drawConnectingArrows(ax, points):
    for idx in range(len(points)):
        ax.arrow(
            points[idx-1][0], points[idx-1][1],
            points[idx][0] - points[idx-1][0], points[idx][1] - points[idx-1][1],
            facecolor=[0, 0, 0, 0.3],
            width=0.02,
            edgecolor='none',
            length_includes_head=True
            )

#tankName =  '#20JUN1-200916-111545_PL' # Good
tankName = '#21JAN5-210622-180202_PL' # wandering

data = parseAllData(tankName)
ParsedData = data['ParsedData']
zoneClass = data['zoneClass']
numTrial = data['numTrial']
midPointTimes = data['midPointTimes']
Trials = data['Trials']
IRs = data['IRs']
AEResult = data['AEResult']

neural_data_transformed, centroids = getZoneLDA(data['neural_data'], zoneClass)
fig, ax = drawLDAResult(neural_data_transformed, zoneClass, centroids, tankName, 200, useOldFigure=True, drawOnlyCloserObjects=True)


isWanderingInNest = np.zeros(neural_data_transformed.shape[0], dtype=bool)
isReadyInNest = np.zeros(neural_data_transformed.shape[0], dtype=bool)

for trial in np.arange(2, numTrial+1):
    latency2HeadEntry = ParsedData[trial-1, 1][0, 0]

    betweenTRON_firstIRON = np.logical_and(
        ParsedData[trial-1, 0][0,0] <= midPointTimes,
        midPointTimes <  (ParsedData[trial-1, 1][0,0] + ParsedData[trial-1, 0][0,0])
    )

    # get behavior types
    if latency2HeadEntry >= 5:
        isWanderingInNest = np.logical_or(isWanderingInNest, np.logical_and(betweenTRON_firstIRON, zoneClass==0))
    else:
        isReadyInNest = np.logical_or(isReadyInNest, np.logical_and(betweenTRON_firstIRON, zoneClass == 0))


centroids['wanderInNest'] = np.mean(neural_data_transformed[isWanderingInNest,:],0)
centroids['readyInNest'] = np.mean(neural_data_transformed[isReadyInNest,:],0)

ax.scatter(centroids['wanderInNest'][0], centroids['wanderInNest'][1], s=100, marker='x', color='g')
ax.scatter(centroids['readyInNest'][0], centroids['readyInNest'][1], s=100, marker='x', color='r')

ax.legend(['Nest', 'Foraging', 'center_Nest', 'center_Foraging', 'center_wander', 'center_ready'])

#
# isStartInNest = np.zeros(neural_data_transformed.shape[0], dtype=bool)
# isRunning2Robot = np.zeros(neural_data_transformed.shape[0], dtype=bool)
# isReturning2Nest = np.zeros(neural_data_transformed.shape[0], dtype=bool)
#
# for trial in np.arange(2, numTrial+1):
#     betweenTROF_TRON = np.logical_and(ParsedData[trial-2, 0][0,1] <= midPointTimes, midPointTimes <  ParsedData[trial-1, 0][0,0])
#     betweenTRON_firstIRON = np.logical_and(
#         ParsedData[trial-1, 0][0,0] <= midPointTimes,
#         midPointTimes <  (ParsedData[trial-1, 1][0,0] + ParsedData[trial-1, 0][0,0])
#     )
#
#     # get behavior types
#     isStartInNest = np.logical_or(isStartInNest, np.logical_and(betweenTRON_firstIRON, zoneClass==0))
#     isRunning2Robot = np.logical_or(isRunning2Robot, np.logical_and(betweenTRON_firstIRON, zoneClass==1))
#     isReturning2Nest = np.logical_or(isReturning2Nest, np.logical_and(betweenTROF_TRON, zoneClass==1))
#
# centroids['startInNest'] = np.mean(neural_data_transformed[isStartInNest,:],0)
# centroids['running2Robot'] = np.mean(neural_data_transformed[isRunning2Robot,:],0)
# centroids['returning2Nest'] = np.mean(neural_data_transformed[isReturning2Nest,:],0)
#
# ax.scatter(centroids['startInNest'][0], centroids['startInNest'][1], color=np.array([85,98,112])/255)
# ax.scatter(centroids['running2Robot'][0], centroids['running2Robot'][1], color=np.array([249,115,6])/255)
# ax.scatter(centroids['returning2Nest'][0], centroids['returning2Nest'][1], color=np.array([85,156,38])/255)
#
# points = [
#         centroids['nest'],
#         centroids['startInNest'],
#         centroids['running2Robot'],
#         centroids['encounter'],
#         centroids['returning2Nest']
#         ]
# drawConnectingArrows(ax, points)
#
# ax.legend(['Nest', 'Foraging', 'Encounter', 'center_Nest', 'center_Foraging', 'center_Encounter', 'start from the nest', 'run to the robot', 'return to the nest'])

# # Hypothesis : if, neural vector during the nesting area is closer to the "state of encounter zone",
# # then the higher chance of avoidance failure on the following trial
# 
# # in this part, 'trial' is actual trial number = There is no 0 Trial
# AvoidData = []
# EscapeData = []
# sumTargetIndex = 0
# for trial in np.arange(2, numTrial+1):
#     betweenTrials = np.logical_and(Trials[trial-2, 1] <= midPointTimes, midPointTimes <  Trials[trial-1, 0])
#     targetIndex = np.logical_and(zoneClass == 0, betweenTrials)
#     print(f'In Nesting During Trial {trial-1} = {np.sum(targetIndex)}')
#     sumTargetIndex += np.sum(targetIndex)
#     if np.sum(targetIndex) == 0:
#         continue
#     distance2encounterState = np.mean(np.sum((centroids['encounter'] - neural_data_transformed[targetIndex,:]) ** 2, 1) ** .5)
#     if AEResult[trial-1] == 0: # current Trial's result is avoid
#         AvoidData.append(distance2encounterState)
#     else:
#         EscapeData.append(distance2encounterState)
