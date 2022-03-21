from pathlib import Path
import numpy as np
from scipy.io import loadmat, savemat

base_path = Path('D:\Data\Lobster\Lobster_Recording-200319-161008\TimeAxisAnalysisDataset')

result = np.zeros((10,15))

f_list = sorted([i for i in base_path.iterdir()])
for i, f in enumerate(f_list):
    if f.is_dir():
        print(f)
        datafile = next(f.glob('output_IL.mat'))
        data = loadmat(str(datafile.absolute()))
        data = data.get('mat_balanced_accuracy')
        result[i,:] = data[:,1]

savemat(base_path / 'result.mat', {'result' : result})