"""
drawLDAGraph
@2022 Knowblesse
2022NOV07
This script is for the testing or drawing one session data.
All the code used in this script is replicated in the `LDABatchTesting.py` for whole session statistics.
"""

import numpy as np
import matplotlib.pyplot as plt
from Switching.SwitchingHelper import parseAllData, getZoneLDA
plt.rcParams["font.family"] = "Noto Sans"
plt.rcParams["font.size"] = 6.6
def drawLDAResult(neural_data_transformed, zoneClass, centroids, tankName, dotNumber=200, drawOnlyCloserObjects=False, useOldFigure=False, points2draw=['n', 'f', 'e']):
    if useOldFigure:
        fig = plt.gcf()
    else:
        fig = plt.figure(figsize=(3.,2.7))
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

    legendText = []
    legendItems = []
    if 'n' in points2draw:
        ax.scatter(neural_data_transformed[zC0, 0], neural_data_transformed[zC0, 1], color=np.array([207,63,66,90])/255, s=7)
        sc = ax.scatter(centroids['nest'][0], centroids['nest'][1], color=np.array([207,63,66])/255, edgecolor='black', marker='D', s=80)
        legendText.append('Nesting')
        #legendText.append('center - Nesting')
        legendItems.append(sc)
    if 'f' in points2draw:
        ax.scatter(neural_data_transformed[zC1, 0], neural_data_transformed[zC1, 1], color=np.array([84,193,223,90])/255, s=7)
        sc = ax.scatter(centroids['foraging'][0], centroids['foraging'][1], color=np.array([84,193,223])/255, edgecolor='black', marker='D', s=80)
        legendText.append('Foraging')
        #legendText.append('center - Foraging')
        legendItems.append(sc)
    if 'e' in points2draw:
        ax.scatter(neural_data_transformed[zC2, 0], neural_data_transformed[zC2, 1], color=np.array([241,136,26,90])/255, s=7)
        sc = ax.scatter(centroids['encounter'][0], centroids['encounter'][1], color=np.array([241,136,26])/255, edgecolor='black', marker='D', s=80)
        legendText.append('Encounter')
        #legendText.append('center - Encounter')
        legendItems.append(sc)
    #ax.legend(legendText)
    ax.legend(legendItems, legendText, fontsize=6.6, markerscale=0.4)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    #ax.set_title(tankName + "- LDA")
    return (fig, ax, legendText, legendItems)
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
#tankName = '#21JAN2-210406-190737_IL'
#tankName = '#21JAN2-210419-175714_IL'

data = parseAllData(tankName)
neural_data = data['neural_data']
ParsedData = data['ParsedData']
zoneClass = data['zoneClass']
numTrial = data['numTrial']
midPointTimes = data['midPointTimes']
Trials = data['Trials']
IRs = data['IRs']
AEResult = data['AEResult']


#####################################################################
#              Draw Example LDA result with scatter                 #
#####################################################################
neural_data_transformed, centroids = getZoneLDA(neural_data, zoneClass)
fig, ax, legendText, legendItems = drawLDAResult(neural_data_transformed, zoneClass, centroids, tankName, 200, useOldFigure=False, drawOnlyCloserObjects=True)


#####################################################################
#                 Get Index for specific behavior                   #
#####################################################################

# isWanderingInNest : staying in the nest zone, between TRON and first IRON, 
# in trials where the latency to the first head entry is more than 5 sec
isWanderingInNest = np.zeros(neural_data_transformed.shape[0], dtype=bool)

# isReadyInNest : staying in the nest zone, between TRON and first IRON, 
# in trials where the latency to the first head entry is shorter than 5 sec
isReadyInNest = np.zeros(neural_data_transformed.shape[0], dtype=bool)

# isStartInNest : staying in the nest zone, between TRON and first IRON
isStartInNest = np.zeros(neural_data_transformed.shape[0], dtype=bool)

# isRunning2Robot : running in the foraging zone, between TRON and first IRON
isRunning2Robot = np.zeros(neural_data_transformed.shape[0], dtype=bool)

# isReturning2Nest : moving in the foraging zone, after Door close (TROF) before new TRON
isReturning2Nest = np.zeros(neural_data_transformed.shape[0], dtype=bool)

# isEncounterAvoid, isEncounterEscape : staying in the encounter zone, between TRON and TROF
isEncounterAvoid = np.zeros(neural_data_transformed.shape[0], dtype=bool)
isEncounterEscape = np.zeros(neural_data_transformed.shape[0], dtype=bool)

for trial in np.arange(1, numTrial): # data from the first trial (trial=0) is ignored
    betweenTRON_TROF = np.logical_and(ParsedData[trial, 0][0,0] <= midPointTimes, midPointTimes <  ParsedData[trial, 0][0,1])
    betweenTROF_TRON = np.logical_and(ParsedData[trial-1, 0][0,1] <= midPointTimes, midPointTimes <  ParsedData[trial, 0][0,0])
    betweenTRON_firstIRON = np.logical_and(
        ParsedData[trial, 0][0,0] <= midPointTimes,
        midPointTimes <  (ParsedData[trial, 1][0,0] + ParsedData[trial, 0][0,0])
    )
    latency2HeadEntry = ParsedData[trial, 1][0, 0]

    # get behavior types
    isStartInNest = np.logical_or(isStartInNest, np.logical_and(betweenTRON_firstIRON, zoneClass==0))
    isRunning2Robot = np.logical_or(isRunning2Robot, np.logical_and(betweenTRON_firstIRON, zoneClass==1))
    isReturning2Nest = np.logical_or(isReturning2Nest, np.logical_and(betweenTROF_TRON, zoneClass==1))

    if latency2HeadEntry >= 5:
        isWanderingInNest = np.logical_or(isWanderingInNest, np.logical_and(betweenTRON_firstIRON, zoneClass==0))
    else:
        isReadyInNest = np.logical_or(isReadyInNest, np.logical_and(betweenTRON_firstIRON, zoneClass == 0))

    if AEResult[trial] == 0:
        isEncounterAvoid = np.logical_or(isEncounterAvoid, np.logical_and(betweenTRON_TROF, zoneClass == 2))
    else:
        isEncounterEscape = np.logical_or(isEncounterEscape, np.logical_and(betweenTRON_TROF, zoneClass == 2))



centroids['wanderInNest'] = np.mean(neural_data_transformed[isWanderingInNest,:],0)
centroids['readyInNest'] = np.mean(neural_data_transformed[isReadyInNest,:],0)
centroids['startInNest'] = np.mean(neural_data_transformed[isStartInNest,:],0)
centroids['running2Robot'] = np.mean(neural_data_transformed[isRunning2Robot,:],0)
centroids['returning2Nest'] = np.mean(neural_data_transformed[isReturning2Nest,:],0)
centroids['encounterAvoid'] = np.mean(neural_data_transformed[isEncounterAvoid,:],0)
centroids['encounterEscape'] = np.mean(neural_data_transformed[isEncounterEscape,:],0)

#####################################################################
#                 Draw Neural State Change Path                     #
#####################################################################
fig2, ax2, legendText, legendItems = drawLDAResult(neural_data_transformed, zoneClass, centroids, tankName, 200, useOldFigure=False, drawOnlyCloserObjects=True)

ax2.scatter(centroids['startInNest'][0], centroids['startInNest'][1], color=np.array([85,98,112])/255)
ax2.scatter(centroids['running2Robot'][0], centroids['running2Robot'][1], color=np.array([249,115,6])/255)
ax2.scatter(centroids['returning2Nest'][0], centroids['returning2Nest'][1], color=np.array([85,156,38])/255)

points = [
        centroids['nest'],
        centroids['startInNest'],
        centroids['running2Robot'],
        centroids['encounter'],
        centroids['returning2Nest']
        ]
drawConnectingArrows(ax2, points)
legendText.extend(['start from the nest', 'run to the robot', 'return to the nest'])
ax2.legend(legendItems, legendText)

#####################################################################
#                 Draw Wander and Ready difference                  #
#####################################################################
fig3, ax3, legendText, legendItems = drawLDAResult(neural_data_transformed, zoneClass, centroids, tankName, 200, useOldFigure=False, drawOnlyCloserObjects=True, points2draw=['n','f'])

sc1 = ax3.scatter(centroids['wanderInNest'][0], centroids['wanderInNest'][1], s=50, marker='^', color='k')
sc2 = ax3.scatter(centroids['readyInNest'][0], centroids['readyInNest'][1], s=50, marker='v', color='k')
legendItems.extend([sc1, sc2])
legendText.extend(['Wander', 'Engaged'])
ax3.legend(legendItems, legendText, fontsize=6.6, markerscale=0.4)


#####################################################################
#                  Draw Avoid vs Escape difference                  #
#####################################################################
fig4, ax4, legendText, legendItems = drawLDAResult(neural_data_transformed, zoneClass, centroids, tankName, 200, useOldFigure=False, drawOnlyCloserObjects=True)
ax4.scatter(centroids['encounterAvoid'][0], centroids['encounterAvoid'][1], s=100, marker='x', color='g')
ax4.scatter(centroids['encounterEscape'][0], centroids['encounterEscape'][1], s=100, marker='x', color='r')
legendText.extend(['center_Avoid', 'center_escape'])
ax4.legend(legendText)
