import numpy as np
from pathlib import Path
import re
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt


## Make All regression raw data into one big file
FolderLocation = Path(r'D:\Data\Lobster\LocationRegression')
OutputFileLocation = Path(r'F:\Output')

OutputData = DataFrame(columns=['Session', 'True_R', 'True_C', 'True_D', 'Fake_R', 'Fake_C', 'Fake_D', 'Pred_R', 'Pred_C', 'Pred_D'])

files = [f for f in FolderLocation.glob('#*')]
for file in files:
    file_name = re.search('#.*',str(file))[0]
    session_name = re.search('#.*r', str(file))[0][:-1]
    FILE_location = file.absolute()
    data = pd.read_csv(FILE_location, sep=',',names=['True_R', 'True_C', 'True_D', 'Fake_R', 'Fake_C', 'Fake_D', 'Pred_R', 'Pred_C', 'Pred_D'])
    data.insert(0,'Session','')
    data['Session'] = session_name
    data['Error_R'] = data['True_R'] - data['Pred_R']
    data['Error_C'] = data['True_C'] - data['Pred_C']
    data['Error_D'] = data['True_D'] - data['Pred_D']
    OutputData= pd.concat([OutputData,data])

OutputData.to_csv(str(OutputFileLocation/'TotalRegression.csv'))


#
data = pd.read_csv(r'F:\Output\TotalRegression.csv',index_col=0)
data = data.sort_values('True_C')

plotdata = np.empty(0)
for i in range(40,620,20):
    index = np.argwhere(
        np.logical_and((data['True_C'].values >= i),
                       (data['True_C'].values < i + 20))
    )
    plotdata = np.append(plotdata,np.mean(np.abs(data['Error_C'].values[index])))

plt.figure(1)
plt.gcf().set_size_inches([5.89, 5.46])
plt.clf()
plt.plot(range(40,620,20), plotdata * 0.169, 'k')
plt.xlim([0,640])
plt.ylim([0,50])
plt.vlines(250,0,50, color='k', linestyles='--')
plt.vlines(525,0,50, color='k', linestyles='--')
plt.xlabel('Location (px)')
plt.ylabel('Error (cm)')
plt.title('Error by Column Location')


data = data.sort_values('True_R')

plotdata = np.empty(0)
for i in range(100,460,20):
    index = np.argwhere(
        np.logical_and((data['True_R'].values >= i),
                       (data['True_R'].values < i + 20))
    )
    plotdata = np.append(plotdata,np.mean(np.abs(data['Error_C'].values[index])))

plt.figure(2)
plt.gcf().set_size_inches([5.89, 5.46])
plt.clf()
plt.plot(range(100,460,20), plotdata*0.169, 'k')
plt.vlines(250,0,50, color='k', linestyles='--')
plt.vlines(315,0,50, color='k', linestyles='--')
plt.xlim([0,480])
plt.ylim([0,40])
plt.xlabel('Location (px)')
plt.ylabel('Error (cm)')
plt.title('Error by Row Location')



## L1 Error Calculation
FolderLocation = Path(r'D:\Data\Lobster\LocationRegression')
OutputFileLocation = Path(r'F:\Output')

OutputData = DataFrame(columns=['Session', 'Error_R', 'Error_C', 'Error_D'])

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
                             'Error_R_Fake':np.mean(np.abs(data['True_R'] - data['Fake_R'])),
                             'Error_C_Fake':np.mean(np.abs(data['True_C'] - data['Fake_C'])),
                             'Error_D_Fake':np.mean(np.abs(data['True_D'] - data['Fake_D'])),
                             'Error_R_True':np.mean(np.abs(data['True_R'] - data['Pred_R'])),
                             'Error_C_True':np.mean(np.abs(data['True_C'] - data['Pred_C'])),
                             'Error_D_True':np.mean(np.abs(data['True_D'] - data['Pred_D'])),})])

OutputData.to_csv(str(OutputFileLocation/'summaryRegression.csv'))