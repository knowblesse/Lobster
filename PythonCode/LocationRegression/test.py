import csv
from scipy.io import loadmat
import re
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import cv2 as cv
from pandas import DataFrame
import pandas as pd

## L1 Error Calculation
#FolderLocation = Path(r'D:\Data\Lobster\LocationRegression')
FolderLocation = Path(r'F:\Output_woNestZone')
OutputFileLocation = Path(r'F:\Output')

OutputData = DataFrame()

files = [f for f in FolderLocation.glob('#*')]
for file in files:
    file_name = re.search('#.*', str(file))[0]
    session_name = re.search('#.*r', str(file))[0][:-1]
    FILE_location = file.absolute()
    data = pd.read_csv(FILE_location, sep=',',
                       names=['True_R', 'True_C', 'True_D', 'Fake_R', 'Fake_C', 'Fake_D', 'Pred_R', 'Pred_C', 'Pred_D'])
    outlierIndex = np.logical_or(
        np.logical_or(data['Pred_R'] < 0, data['Pred_R'] > 480),
        np.logical_or(data['Pred_C'] < 0, data['Pred_C'] > 640)
    )
    data = data.drop(index=np.where(outlierIndex)[0])
    OutputData = pd.concat([OutputData,
                            DataFrame({'Session': session_name,
                                       'Error_R_Fake': 0.169*np.mean(np.abs(data['True_R'] - data['Fake_R'])),
                                       'Error_C_Fake': 0.169*np.mean(np.abs(data['True_C'] - data['Fake_C'])),
                                       'Error_D_Fake': np.mean(np.abs(data['True_D'] - data['Fake_D'])),
                                       'Error_R_True': 0.169*np.mean(np.abs(data['True_R'] - data['Pred_R'])),
                                       'Error_C_True': 0.169*np.mean(np.abs(data['True_C'] - data['Pred_C'])),
                                       'Error_D_True': np.mean(np.abs(data['True_D'] - data['Pred_D']))
                                                 }, index=[0])
                            ])

OutputData.to_csv(str(OutputFileLocation / 'summaryRegression_woNestZone.csv'))






##################################
from NeuralPatternSwitch import wholeSessionUnitDataPCA

# Batch
import re
from pathlib import Path
import matplotlib.pyplot as plt

FolderPath = Path(r'F:\LobsterData')

for tank in FolderPath.glob('#*'):
    tank_name = re.search('#.*', str(tank))[0]
    TANK_location = tank.absolute()

    # Butter location
    butter_location = [p for p in TANK_location.glob('*_buttered.csv')]

    if len(butter_location) == 0:
        raise (BaseException("Can not find a butter file in the current Tank location"))
    elif len(butter_location) > 1:
        raise (BaseException("There are multiple files ending with _buttered.csv"))

    # Check if the neural data file is present
    wholeSessionUnitData_location = [p for p in TANK_location.glob('*_wholeSessionUnitData.csv')]

    if len(wholeSessionUnitData_location) == 0:
        raise (BaseException("Can not find a regression data file in the current Tank location"))
    elif len(wholeSessionUnitData_location) > 1:
        raise (BaseException("There are multiple files ending with _wholeSessionUnitData.csv"))

    # Main Script
    wholeSessionUnitDataPCA(Tank_Path=TANK_location, n_cluster=2)
    plt.pause(2)





