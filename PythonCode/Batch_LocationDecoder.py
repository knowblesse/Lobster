"""
Batch_Location Decoder
"""
import numpy as np
from pathlib import Path
from LocationDecoder import LocationDecoder

FolderLocation = Path(r'D:\Data\Lobster\Lobster_Recording-200319-161008\20JUN1')
OutputFileLocation = Path(r'D:\Output')

for tank in FolderLocation.glob('#*'):
    tank_name = re.search('#.*', str(tank))[0]
    tank_name h
    WholeTestResult = LocationDecoder(tank)
    np.savetxt(str(OutputFileLocation / (tank_name + 'result.csv')),WholeTestResult, fmt='%.3f', delimiter=',')

