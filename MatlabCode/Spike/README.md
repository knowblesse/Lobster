## AlignEvent
Align spike data on a specific event, compute Z-score and save into **aligned** folder


## drawPeakSortedPETH
**function**

`drawPeakSortedPETH(zscores, TIMEWINDOW, TIMEWINDOW_BIN, ax_hm, ax_hist, options)`

Draw peak sorted PETH with mean zscore data around the onset of the event.

Used to draw figure2

## drawPETH
**function**

`ax = drawPETH(unit, TIMEWINDOW, ax_raster, ax_histo, normalize)`

Draw PETH of one unit.

For drawing representative unit activity figure.

## getTimepointFromParsedData
**function**

`[timepoint,numTrial] = getTimepointFromParsedData(ParsedData,drawOnly)`

Find absolute time of the each behavior event.

Using the ParsedData, this function output timepoint struct variable which has all behavioral absolute time point for every trials. Timepoines are in ms.

drawOnly : 'A' or 'E'. if provided, return only Avoid or Escape trials


## loadAlignedData
**function**

Load Aligned unit data (.mat) and make them into one cell

## loadAllUnitData
**function**

Load all unit data with corresponding behavior data

## loadUnitData
**function**

Load unit data (.mat) into path variable

## PETH_Scripts.m
Draw PETH of one unit.

For drawing representative unit activity figure.

## plotMeanWaveform.m
Plot mean waveform from SU variable

## SortedPETH_Scripts
Main script for analyzing and drawing figures in Fig2

## unitstxt2mats.m
**function**

Converts a txt file exported from the Offline Sorter into multiple .mat files
