import re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from Switching.SwitchingHelper import parseAllData, getZoneLDA
from sklearn.decomposition import PCA
FolderPath = Path(r'D:/Data/Lobster/FineDistanceDataset')
outputData = []
pbar = tqdm([p for p in FolderPath.glob('#*')])
for tank in pbar:
    array2append = []
    tankName = re.search('#.*', str(tank))[0]
    array2append.append(tankName)

    data = parseAllData(tankName)
    zoneClass = data['zoneClass']
    numTrial = data['numTrial']
    midPointTimes = data['midPointTimes']
    Trials = data['Trials']
    AEResult = data['AEResult']
    ParsedData = data['ParsedData']

    neural_data_transformed, centroids = getZoneLDA(data['neural_data'], zoneClass)

    # Compare within-zone distance vs between-zone distance
    NN = np.mean(np.sum((centroids['nest'] - neural_data_transformed[zoneClass == 0, :])**2, axis=1)**0.5)
    NO = np.mean(
        [np.mean(np.sum((centroids['foraging'] - neural_data_transformed[zoneClass == 0, :]) ** 2, axis=1) ** 0.5),
        np.mean(np.sum((centroids['encounter'] - neural_data_transformed[zoneClass == 0, :]) ** 2, axis=1) ** 0.5)]
    )

    FF = np.mean(np.sum((centroids['foraging'] - neural_data_transformed[zoneClass == 1, :]) ** 2, axis=1) ** 0.5)
    FO = np.mean(
        [np.mean(np.sum((centroids['nest'] - neural_data_transformed[zoneClass == 1, :]) ** 2, axis=1) ** 0.5),
        np.mean(np.sum((centroids['encounter'] - neural_data_transformed[zoneClass == 1, :]) ** 2, axis=1) ** 0.5)]
    )
    F_std = np.mean(np.std(neural_data_transformed[zoneClass == 1, :], axis=0))

    EE = np.mean(np.sum((centroids['encounter'] - neural_data_transformed[zoneClass == 2, :]) ** 2, axis=1) ** 0.5)
    EO = np.mean(
        [np.mean(np.sum((centroids['nest'] - neural_data_transformed[zoneClass == 2, :]) ** 2, axis=1) ** 0.5),
        np.mean(np.sum((centroids['foraging'] - neural_data_transformed[zoneClass == 2, :]) ** 2, axis=1) ** 0.5)]
    )
    array2append.extend([
        NN, NO,
        FF, FO,
        EE, EO
    ])

    # Compare centroid distances
    D_nest_foraging = np.sum((np.array(centroids['nest']) - np.array(centroids['foraging'])) ** 2) ** 0.5
    D_nest_encounter = np.sum((np.array(centroids['nest']) - np.array(centroids['encounter'])) ** 2) ** 0.5
    D_encounter_foraging = np.sum((np.array(centroids['encounter']) - np.array(centroids['foraging'])) ** 2) ** 0.5
    array2append.extend([D_nest_foraging, D_nest_encounter, D_encounter_foraging])

    # Compare Wander vs Engaged
    isWanderingInNest = np.zeros(neural_data_transformed.shape[0], dtype=bool)
    isReadyInNest = np.zeros(neural_data_transformed.shape[0], dtype=bool)

    for trial in np.arange(1, numTrial):
        latency2HeadEntry = ParsedData[trial, 1][0, 0] # first IRON from TRON

        betweenTRON_firstIRON = np.logical_and(
            ParsedData[trial, 0][0, 0] <= midPointTimes,
            midPointTimes < (ParsedData[trial, 1][0, 0] + ParsedData[trial, 0][0, 0])
        )

        # get behavior types
        if latency2HeadEntry >= 5:
            isWanderingInNest = np.logical_or(isWanderingInNest, np.logical_and(betweenTRON_firstIRON, zoneClass == 0))
        else:
            isReadyInNest = np.logical_or(isReadyInNest, np.logical_and(betweenTRON_firstIRON, zoneClass == 0))

    centroids['wanderInNest'] = np.mean(neural_data_transformed[isWanderingInNest, :], 0)
    centroids['readyInNest'] = np.mean(neural_data_transformed[isReadyInNest, :], 0)

    distance_between_wander_c_foraging = np.mean(np.sum((centroids['foraging'] - neural_data_transformed[isWanderingInNest, :]) ** 2, axis=1) ** 0.5)
    distance_between_ready_c_foraging = np.mean(np.sum((centroids['foraging'] - neural_data_transformed[isReadyInNest, :]) ** 2, axis=1) ** 0.5)

    array2append.extend([distance_between_wander_c_foraging, distance_between_ready_c_foraging])
    ####################################################################################################################
    # PCA
    ####################################################################################################################
    pca = PCA()
    neural_data_transformed_pca = pca.fit_transform(data['neural_data'])

    # Calculate Centroids
    centroids_pca = {
        'nest': np.mean(neural_data_transformed_pca[zoneClass == 0, :],0),
        'foraging': np.mean(neural_data_transformed_pca[zoneClass == 1, :],0),
        'encounter': np.mean(neural_data_transformed_pca[zoneClass == 2, :],0)}

    # Compare centroid distances
    D_nest_foraging_pca = np.sum((centroids_pca['nest'] - centroids_pca['foraging']) ** 2) ** 0.5
    D_nest_encounter_pca = np.sum((centroids_pca['nest'] - centroids_pca['encounter']) ** 2) ** 0.5
    D_encounter_foraging_pca = np.sum((centroids_pca['encounter'] - centroids_pca['foraging']) ** 2) ** 0.5
    array2append.extend([D_nest_foraging_pca, D_nest_encounter_pca, D_encounter_foraging_pca])

    # Compare within-zone distance vs between-zone distance
    NN = np.mean(np.sum((centroids_pca['nest'] - neural_data_transformed_pca[zoneClass == 0, :]) ** 2, axis=1) ** 0.5)
    NO = np.mean(
        [np.mean(np.sum((centroids_pca['foraging'] - neural_data_transformed_pca[zoneClass == 0, :]) ** 2, axis=1) ** 0.5),
         np.mean(np.sum((centroids_pca['encounter'] - neural_data_transformed_pca[zoneClass == 0, :]) ** 2, axis=1) ** 0.5)]
    )

    FF = np.mean(np.sum((centroids_pca['foraging'] - neural_data_transformed_pca[zoneClass == 1, :]) ** 2, axis=1) ** 0.5)
    FO = np.mean(
        [np.mean(np.sum((centroids_pca['nest'] - neural_data_transformed_pca[zoneClass == 1, :]) ** 2, axis=1) ** 0.5),
         np.mean(np.sum((centroids_pca['encounter'] - neural_data_transformed_pca[zoneClass == 1, :]) ** 2, axis=1) ** 0.5)]
    )
    F_std = np.mean(np.std(neural_data_transformed[zoneClass == 1, :], axis=0))

    EE = np.mean(np.sum((centroids_pca['encounter'] - neural_data_transformed_pca[zoneClass == 2, :]) ** 2, axis=1) ** 0.5)
    EO = np.mean(
        [np.mean(np.sum((centroids_pca['nest'] - neural_data_transformed_pca[zoneClass == 2, :]) ** 2, axis=1) ** 0.5),
         np.mean(np.sum((centroids_pca['foraging'] - neural_data_transformed_pca[zoneClass == 2, :]) ** 2, axis=1) ** 0.5)]
    )
    array2append.extend([
        NN, NO,
        FF, FO,
        EE, EO
    ])

    # Hypothesis : if, neural vector during the nesting area is closer to the "state of encounter zone",
    # then the higher chance of avoidance failure on the following trial

    # # in this part, 'trial' is actual trial number = There is no 0 Trial
    # AvoidData = []
    # EscapeData = []
    # for trial in np.arange(2, numTrial + 1):
    #     betweenTrials = np.logical_and(Trials[trial - 2, 1] <= midPointTimes, midPointTimes < Trials[trial - 1, 0])
    #     targetIndex = np.logical_and(zoneClass == 0, betweenTrials)
    #     if np.sum(targetIndex) == 0:
    #         continue
    #     distance2encounterState = np.mean(
    #         np.sum((centroids['encounter'] - neural_data_transformed[targetIndex, :]) ** 2, 1) ** .5)
    #     distance2nestState = np.mean(
    #         np.sum((centroids['nest'] - neural_data_transformed[targetIndex, :]) ** 2, 1) ** .5)
    #     vec = (distance2encounterState - distance2nestState) / distance2nestState
    #     if AEResult[trial - 1] == 0:  # current Trial's result is avoid
    #         AvoidData.append(vec)
    #     else:
    #         EscapeData.append(vec)
    # if (len(AvoidData) > 5) and (len(EscapeData) > 5):
    #     outputData.append([tankName, np.mean(AvoidData), np.mean(EscapeData)])
    #
    # # Hypothesis : if, neural vector during the nesting area is closer to the "state of encounter zone",
    # # then the higher chance of avoidance failure on the following trial
    #
    # # # in this part, 'trial' is actual trial number = There is no 0 Trial
    # AvoidData = []
    # EscapeData = []
    # for trial in np.arange(2, numTrial + 1):
    #     betweenTrials = np.logical_and(Trials[trial - 2, 1] <= midPointTimes, midPointTimes < Trials[trial - 1, 0])
    #     targetIndex = np.logical_and(zoneClass == 0, betweenTrials)
    #     if np.sum(targetIndex) == 0:
    #         continue
    #     distance2encounterState = np.mean(
    #         np.sum((centroids['encounter'] - neural_data_transformed[targetIndex, :]) ** 2, 1) ** .5)
    #     distance2nestState = np.mean(
    #         np.sum((centroids['nest'] - neural_data_transformed[targetIndex, :]) ** 2, 1) ** .5)
    #     vec = (distance2encounterState - distance2nestState) / distance2nestState
    #     if AEResult[trial - 1] == 0:  # current Trial's result is avoid
    #         AvoidData.append(vec)
    #     else:
    #         EscapeData.append(vec)
    # if (len(AvoidData) > 5) and (len(EscapeData) > 5):
    #     outputData.append([tankName, np.mean(AvoidData), np.mean(EscapeData)])

    outputData.append(array2append)
np.savetxt('C:/Users/Knowblesse/Desktop/distances.csv', np.array(outputData), delimiter=',', fmt='%s')
