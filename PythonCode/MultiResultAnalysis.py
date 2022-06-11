import numpy as np
from pathlib import Path
import re
import pandas as pd
from pandas import DataFrame

FolderLocation = Path(r'D:\Data\Lobster\LocationRegression')
OutputFileLocation = Path(r'F:\Output')

OutputData = DataFrame(columns=['True_R', 'True_C', 'True_D', 'Fake_R', 'Fake_C', 'Fake_D', 'Pred_R', 'Pred_C', 'Pred_D'])

files = [f for f in FolderLocation.glob('#*')]
for file in files:
    file_name = re.search('#.*',str(file))[0]
    FILE_location = file.absolute()
    pd.read_csv()
    data = pd.read_csv(FILE_location, sep=',',names=['True_R', 'True_C', 'True_D', 'Fake_R', 'Fake_C', 'Fake_D', 'Pred_R', 'Pred_C', 'Pred_D'])

    OutputData= pd.concat([OutputData,data])

OutputData.to_csv(str(OutputFileLocation/'AllData.csv'))


#

data = pd.read_csv(r'F:/Output/AllData.csv')
data = data.sort_values('True_C')

plotdata = np.empty(0)
for i in range(40,620,20):
    index = np.argwhere(
        np.logical_and((data['True_C'].values >= i),
                       (data['True_C'].values < i + 20))
    )
    plotdata = np.append(plotdata,np.mean(data['Error_C'].values[index]))


112 ~ 455.5

data = pd.read_csv(r'F:/Output/AllData.csv')
data = data.sort_values('True_R')

plotdata = np.empty(0)
for i in range(100,460,20):
    index = np.argwhere(
        np.logical_and((data['True_R'].values >= i),
                       (data['True_R'].values < i + 20))
    )
    plotdata = np.append(plotdata,np.mean(data['Error_C'].values[index]))

plt.plot(plotdata)