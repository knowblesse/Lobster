import numpy as np
from pathlib import Path
import re
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


## Make All regression raw data into one big file
FolderLocation = Path(r'D:\Data\Lobster\LocationRegression')
OutputFileLocation = Path(r'F:\Output')

OutputData = DataFrame(columns=['Session', 'Speed', 'True_R', 'True_C', 'True_D', 'Fake_R', 'Fake_C', 'Fake_D', 'Pred_R', 'Pred_C', 'Pred_D'])

files = [f for f in FolderLocation.glob('#*')]
for file in files:
    file_name = re.search('#.*',str(file))[0]
    session_name = re.search('#.*r', str(file))[0][:-1]
    FILE_location = file.absolute()
    data = pd.read_csv(FILE_location, sep=',',names=['True_R', 'True_C', 'True_D', 'Fake_R', 'Fake_C', 'Fake_D', 'Pred_R', 'Pred_C', 'Pred_D'])

    # Calculate the Error
    data.insert(0, 'Session', '')
    data['Session'] = session_name
    data['Error_R'] = data['True_R'] - data['Pred_R']
    data['Error_C'] = data['True_C'] - data['Pred_C']
    data['Error_D'] = data['True_D'] - data['Pred_D']

    # Calculate the speed
    speed = (np.diff(data['True_R'])**2 + np.diff(data['True_C'])**2)**0.5
    data['Speed'] = np.append([0], speed)

    # Remove Outlier
    outlierIndex = np.logical_or(
        np.logical_or(data['Pred_R'] < 0, data['Pred_R'] > 480),
        np.logical_or(data['Pred_C'] < 0, data['Pred_C'] > 640)
    )

    data = data.drop(index=np.where(outlierIndex)[0])
    if 0 in data.index:
        data = data.drop(index=[0])

    OutputData= pd.concat([OutputData,data])

OutputData.to_csv(str(OutputFileLocation/'TotalRegression_withSpeed.csv'))