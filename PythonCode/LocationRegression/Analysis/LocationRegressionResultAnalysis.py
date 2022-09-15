import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt

'''
Location Regression Data : [
    Real Row px, 
    Real Col px,
    Real Deg,
    Shuffled Row px,
    Shuffled Col px,
    Shuffled Deg, 
    Predicted Row px,
    Predicted Col px,
    Predicted Deg
    ]
Distance Regression Data : [Real px, Shuffled px, Predicted px]
'''

## Load All Regression Data
LocationFolderLocation = Path(r'D:\Data\Lobster\LocationRegressionResult')
DistanceFolderLocation = Path(r'D:\Data\Lobster\DistanceRegressionResult')

SessionName = []
LocData = []
DistData = []

for file in sorted([f for f in LocationFolderLocation.glob('#*')]):
    FILE_location = file.absolute()
    session_name = re.search('#.{6}-.{6}-.{6}', str(file))[0]
    data = np.loadtxt(str(FILE_location), delimiter=',')

    data[:, 0:2] = data[:, 0:2] * 0.169
    data[:, 3:5] = data[:, 3:5] * 0.169
    data[:, 6:8] = data[:, 6:8] * 0.169

    SessionName.append(session_name)
    LocData.append(data)

for file in sorted([f for f in DistanceFolderLocation.glob('#*')]):
    FILE_location = file.absolute()
    data = np.loadtxt(str(FILE_location), delimiter=',')

    data = data * 0.169

    DistData.append(data)

## Calculate mean L1 error for Fake and Pred dataset - Location (XY)
OutputFilePath = Path(r'D:\Data\Lobster\LocationRegressionResult')
SummaryData = np.empty((0,6))
with open(OutputFilePath / 'SummaryData.csv','w') as f:
    for session, data in zip(SessionName, LocData):
        error_degree_fake = np.zeros((data.shape[0]))
        error_degree_pred = np.zeros((data.shape[0]))
        for i in range(data.shape[0]):
            s_ = np.sort((data[i, 2], data[i, 5]))
            error_degree_fake[i] = np.min((s_[1] - s_[0], s_[0] + 360-s_[1]))
            s_ = np.sort((data[i, 2], data[i, 8]))
            error_degree_pred[i] = np.min((s_[1] - s_[0], s_[0] + 360 - s_[1]))
        f.writelines(f'{session}, '
                     f'{np.mean(np.abs(data[:,0] - data[:,3])):.2f}, {np.mean(np.abs(data[:,0] - data[:,6])):.2f}, ' # Row shuffled Row Real
                     f'{np.mean(np.abs(data[:,1] - data[:,4])):.2f}, {np.mean(np.abs(data[:,1] - data[:,7])):.2f}, '
                     f'{np.mean(error_degree_fake):.2f}, {np.mean(error_degree_pred):.2f}\n')

## Calculate mean L1 error for Fake and Pred dataset - Distance
OutputFilePath = Path(r'D:\Data\Lobster\DistanceRegressionResult')
SummaryData = np.empty((0,2))
with open(OutputFilePath / 'SummaryData.csv','w') as f:
    for session, data in zip(SessionName, DistData):
        f.writelines(f'{session}, {np.mean(np.abs(data[:,0] - data[:,1])):.2f}, {np.mean(np.abs(data[:,0] - data[:,2])):.2f}\n')

##

for data in RData:
    print(np.max(data[]))


OutputData.to_csv(str(OutputFileLocation/'TotalRegression_Control.csv'))


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
plt.plot(range(40,620,20), plotdata * 0.169, 'g')
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
plt.plot(range(100,460,20), plotdata*0.169, 'g')
plt.vlines(250,0,50, color='k', linestyles='--')
plt.vlines(315,0,50, color='k', linestyles='--')
plt.xlim([0,480])
plt.ylim([0,40])
plt.xlabel('Location (px)')
plt.ylabel('Error (cm)')
plt.title('Error by Row Location')



## L1 Error Calculation
FolderLocation = Path(r'F:\Output_Control')
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

OutputData.to_csv(str(OutputFileLocation/'summaryRegression_Control.csv'))