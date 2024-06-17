#!/usr/bin/env python3

import os, sys
import pandas as pd

bitDepth = '${params.bit_depth}'
def findMissingFeatures(df, nom, designFile):
	panelDesign = pd.read_csv(designFile)
	print(panelDesign)



if __name__ == "__main__":
	myDataFile = pd.read_pickle("${pickleTable}")
	myFileIdx = "${batchID}"
	panelCsvFile = "${designTable}"

	findMissingFeatures(myDataFile, myFileIdx, panelCsvFile)


