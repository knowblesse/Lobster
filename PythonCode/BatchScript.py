"""
BatchScript
@ 2022 Knowblesse
BatchScript for Running through all Session Data
"""



# Method 1 : All Tanks are in a single folder

import re
from pathlib import Path


FolderPath = Path(r'F:\LobsterData')

for tank in FolderPath.glob('#*'):
    tank_name = re.search('#.*', str(tank))[0]
    TANK_location = tank.absolute()
    
    # Butter location
    butter_location = [p for p in TANK_location.glob('*_buttered.csv')]

    if len(butter_location) == 0:
        raise(BaseException("Can not find a butter file in the current Tank location"))
    elif len(butter_location) > 1:
        raise(BaseException("There are multiple files ending with _buttered.csv"))

    # Check if the neural data file is present
    wholeSessionUnitData_location = [p for p in TANK_location.glob('*_wholeSessionUnitData.csv')]

    if len(wholeSessionUnitData_location) == 0:
        raise(BaseException("Can not find a regression data file in the current Tank location"))
    elif len(wholeSessionUnitData_location) > 1:
        raise(BaseException("There are multiple files ending with _wholeSessionUnitData.csv"))

    # Main Script

