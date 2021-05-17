## AlignEvent.m
Align spike data on a specific event, compute Z-score and save into **aligned** folder

## deleteAlignedData.m
(temporal script)
Delete all **Aligned** folders from rawdata tank

## DeleteUnitMat.m
(tempral script)
Delete all **unit.mat** files from rawdata tank's recording folder

## drawPETH.m
### ax = drawPETH(unit, TIMEWINDOW)
Draw Peri-Event Time Histogram and returns two axes in a cell structure. 
- unit : cell : nx1 : each cell represent one trial and it has n x 1 matrix

## DrawPETHforAllUnits.m
Batch script for **drawPETH.m**

## FiringRateCalculator.m

## getTimepointFromParsedData.m

## loadAlignedData.m
Load Aligned unit data (.mat) and make them into one cell

## loadUnitData.m
Load unit data (.mat) into path variable

## plotMeanWaveform.m
(temporal script)
Plot mean waveform from SU variable

## subAlignEvent_separateAE.m
Subscript of **AlignEvent.m**
Divide trials into Avoid and Escape, align and calculate Z-score

## unitstxt2mats.m
Converts a txt file exported from the Offline Sorter into multiple .mat files
