from pathlib import Path
from pandas import DataFrame
import pandas as pd
import numpy as np
import re

## L1 Error Calculation
FolderLocation = Path(r'F:\Output_Control')
OutputFileLocation = Path(r'F:\Output')

OutputData = DataFrame(columns=['Session', 'Error_R_Fake', 'Error_R_True', 'Error_C_Fake', 'Error_C_True', 'Error_D_Fake', 'Error_D_True'])

files = [f for f in FolderLocation.glob('#*')]
for file in files:
    file_name = re.search('#.*',str(file))[0]
    session_name = re.search('#.*r', str(file))[0][:-1]
    FILE_location = file.absolute()
    data = pd.read_csv(FILE_location, sep=',',names=['True_R', 'True_C', 'True_D', 'Fake_R', 'Fake_C', 'Fake_D', 'Pred_R', 'Pred_C', 'Pred_D'])
    outlierIndex = np.logical_or(
        np.logical_or(data['Pred_R'] < 0, data['Pred_R'] > 480),
        np.logical_or(data['Pred_C'] < 0, data['Pred_C'] > 640)
    )
    data = data.drop(index=np.where(outlierIndex)[0])
    OutputData = pd.concat([OutputData,
                            DataFrame({'Session':session_name,
                             'Error_R_Fake': np.mean(np.abs(data['True_R'] - data['Fake_R'])) * 0.169,
                             'Error_R_True': np.mean(np.abs(data['True_R'] - data['Pred_R'])) * 0.169,
                             'Error_C_Fake': np.mean(np.abs(data['True_C'] - data['Fake_C'])) * 0.169,
                             'Error_C_True': np.mean(np.abs(data['True_C'] - data['Pred_C'])) * 0.169,
                             'Error_D_Fake': np.mean(np.abs(data['True_D'] - data['Fake_D'])),
                             'Error_D_True': np.mean(np.abs(data['True_D'] - data['Pred_D']))}, index=[0])])

OutputData.to_csv(str(OutputFileLocation/'summaryRegression_Control.csv'))