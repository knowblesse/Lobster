"""
LocationRegressionResultAnalysis
@ Knowblesse 2022
22SEP29
Parsing Location Regression Dataset.
Further processing (mostly plotting) is done in the matlab
"""
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt

'''
Location Regression Data : [
    0 : Real Row px, 
    1 : Real Col px,
    2 : Real Deg,
    3 : Shuffled Row px,
    4 : Shuffled Col px,
    5 : Shuffled Deg, 
    6 : Predicted Row px,
    7 : Predicted Col px,
    8 : Predicted Deg
    ]
Distance Regression Data : [Real px, Shuffled px, Predicted px]
'''

#############################################
#          Load All Regression Data         #
#############################################
LocationFolderLocation = Path(r'D:\Data\Lobster\LocationRegressionResult')
DistanceFolderLocation = Path(r'D:\Data\Lobster\DistanceRegressionResult')
px2cm = 0.169

SessionName = []
LocData = []
DistData = []

# Load LocationData
for file in sorted([f for f in LocationFolderLocation.glob('#*')]):
    FILE_location = file.absolute()
    session_name = re.search('#.{6}-.{6}-.{6}', str(file))[0]
    data = np.loadtxt(str(FILE_location), delimiter=',')
    SessionName.append(session_name)
    LocData.append(data)

# Load DistanceData
for file in sorted([f for f in DistanceFolderLocation.glob('#*')]):
    FILE_location = file.absolute()
    data = np.loadtxt(str(FILE_location), delimiter=',')
    DistData.append(data)

#############################################
#        Calculate L1 error - Location      #
#############################################
def getDegreeError(degree1, degree2):
    sorted_degree = np.sort((degree1, degree2))
    return np.min( (sorted_degree[1] - sorted_degree[0], 
     
for session, data in zip(SessionName, LocData):
    error_row_fake = np.mean(np.abs(data[:,0] - data[:,3]))
    error_row_pred = np.mean(np.abs(data[:,0] - data[:,6]))
    error_col_fake = np.mean(np.abs(data[:,1] - data[:,4]))
    error_col_pred = np.mean(np.abs(data[:,1] - data[:,7]))
    
    # calculate error 
    error_degree_fake = np.zeros((data.shape[0]))
    error_degree_pred = np.zeros((data.shape[0]))
    for i in range(data.shape[0]):
        sorted_degree = np.sort((data[i, 2], data[i, 5]))
        error_degree_fake[i] = np.min((sorted_degree[1] - sorted_degree[0], sorted_degree[0] + 360-sorted_degree[1]))
        sorted_degree = np.sort((data[i, 2], data[i, 8]))
        error_degree_pred[i] = np.min((sorted_degree[1] - sorted_degree[0], sorted_degree[0] + 360 - sorted_degree[1]))
    f.writelines(f'{session}, '
                 f'{:.2f}, {:.2f}, ' # Row shuffled Row Real
                 f'{:.2f}, {:.2f}, '
                 f'{np.mean(error_degree_fake):.2f}, {np.mean(error_degree_pred):.2f}\n')

#############################################
#        Calculate L1 error - Distnace      #
#############################################
OutputFilePath = Path(r'D:\Data\Lobster\DistanceRegressionResult')
SummaryData = np.empty((0,2))
with open(OutputFilePath / 'SummaryData.csv','w') as f:
    for session, data in zip(SessionName, DistData):
        f.writelines(f'{session}, {np.mean(np.abs(data[:,0] - data[:,1])):.2f}, {np.mean(np.abs(data[:,0] - data[:,2])):.2f}\n')

## Calculte Mean L1 error for Fake and Pred dataset - Distance - Exclude Nesting zone
OutputFilePath = Path(r'D:\Data\Lobster\DistanceRegressionResult')
SummaryData = np.empty((0,2))
with open(OutputFilePath / 'SummaryData_woNesting.csv','w') as f:
    for session, locData, distData in zip(SessionName, LocData, DistData):
        # Check if the rat is in the Foragning zone
        isForagining = (locData[:,1] / 0.169) > 255
        L1_error_shuffled = np.abs(distData[:, 0] - distData[:, 1])
        L1_error_real = np.abs(distData[:, 0] - distData[:, 2])
        f.writelines(f'{session}, {np.mean(L1_error_shuffled[isForagining]):.2f}, {np.mean(L1_error_real[isForagining]):.2f}\n')



#
#
# for data in RData:
#     print(np.max(data[]))
#
#
# OutputData.to_csv(str(OutputFileLocation/'TotalRegression_Control.csv'))
#
#
# #
# data = pd.read_csv(r'F:\Output\TotalRegression.csv',index_col=0)
# data = data.sort_values('True_C')
#
# plotdata = np.empty(0)
# for i in range(40,620,20):
#     index = np.argwhere(
#         np.logical_and((data['True_C'].values >= i),
#                        (data['True_C'].values < i + 20))
#     )
#     plotdata = np.append(plotdata,np.mean(np.abs(data['Error_C'].values[index])))
#
# plt.figure(1)
# plt.gcf().set_size_inches([5.89, 5.46])
# plt.clf()
# plt.plot(range(40,620,20), plotdata * 0.169, 'g')
# plt.xlim([0,640])
# plt.ylim([0,50])
# plt.vlines(250,0,50, color='k', linestyles='--')
# plt.vlines(525,0,50, color='k', linestyles='--')
# plt.xlabel('Location (px)')
# plt.ylabel('Error (cm)')
# plt.title('Error by Column Location')
#
#
# data = data.sort_values('True_R')
#
# plotdata = np.empty(0)
# for i in range(100,460,20):
#     index = np.argwhere(
#         np.logical_and((data['True_R'].values >= i),
#                        (data['True_R'].values < i + 20))
#     )
#     plotdata = np.append(plotdata,np.mean(np.abs(data['Error_C'].values[index])))
#
# plt.figure(2)
# plt.gcf().set_size_inches([5.89, 5.46])
# plt.clf()
# plt.plot(range(100,460,20), plotdata*0.169, 'g')
# plt.vlines(250,0,50, color='k', linestyles='--')
# plt.vlines(315,0,50, color='k', linestyles='--')
# plt.xlim([0,480])
# plt.ylim([0,40])
# plt.xlabel('Location (px)')
# plt.ylabel('Error (cm)')
# plt.title('Error by Row Location')
#
#
#
# ## L1 Error Calculation
# FolderLocation = Path(r'F:\Output_Control')
# OutputFileLocation = Path(r'F:\Output')
#
# OutputData = DataFrame(columns=['Session', 'Error_R', 'Error_C', 'Error_D'])
#
# files = [f for f in FolderLocation.glob('#*')]
# for file in files:
#     file_name = re.search('#.*',str(file))[0]
#     session_name = re.search('#.*r', str(file))[0][:-1]
#     FILE_location = file.absolute()
#     data = pd.read_csv(FILE_location, sep=',',names=['True_R', 'True_C', 'True_D', 'Fake_R', 'Fake_C', 'Fake_D', 'Pred_R', 'Pred_C', 'Pred_D'])
#     outlierIndex = np.logical_or(
#         np.logical_or(data['Pred_R'] < 0, data['Pred_R'] > 480),
#         np.logical_or(data['Pred_C'] < 0, data['Pred_C'] > 640)
#     )
#     data = data.drop(index=np.where(outlierIndex)[0])
#     OutputData = pd.concat([OutputData,
#                             DataFrame({'Session':session_name,
#                              'Error_R_Fake':np.mean(np.abs(data['True_R'] - data['Fake_R'])),
#                              'Error_C_Fake':np.mean(np.abs(data['True_C'] - data['Fake_C'])),
#                              'Error_D_Fake':np.mean(np.abs(data['True_D'] - data['Fake_D'])),
#                              'Error_R_True':np.mean(np.abs(data['True_R'] - data['Pred_R'])),
#                              'Error_C_True':np.mean(np.abs(data['True_C'] - data['Pred_C'])),
#                              'Error_D_True':np.mean(np.abs(data['True_D'] - data['Pred_D'])),})])
#
# OutputData.to_csv(str(OutputFileLocation/'summaryRegression_Control.csv'))
