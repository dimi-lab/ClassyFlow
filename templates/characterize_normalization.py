#!/usr/bin/env python3

import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf

#objtype = '${params.qupath_object_type}'
objtype = 'CellObject'

def create_comparative_report(pdfOUT, nom, hshOfDFs):
	print("In PDF Making")

	#for ky, df in hshOfDFs.items():
	# myDataFile['DAPI: Cell: Mean'].plot.hist(bins=16)


def select_best_normalization():
	print("TBD")
	# https://cran.r-project.org/web/packages/bestNormalize/vignettes/bestNormalize.html
	#  Selecting the best technique
	# The bestNormalize function selects the best transformation according to the Pearson P statistic (divided by its degrees of freedom), as
	# calculated by the nortest package. There are a variety of normality tests out there, but the benefit of the Pearson P / df is that it is a
	# relatively interpretable goodness of fit test, and the ratio P / df can be compared between transformations as an absolute measure of the 
	# departure from normality (if the data follows close to a normal distribution, this ratio will be close to 1). The transformation whose
	# transformed values fit normality the closest according to this statistic (or equivalently, this ratio), is selected by bestNormalize. The
	# ratios are printed when the object is printed.
	return 'NA'


def buildDataDictionary(lstOfFiles):
	hashOfNormalizationTables = {}
	for fh in lstOfFiles:
		filename, file_extension = os.path.splitext(fh)
		print([filename, file_extension])
		pnom = filename.split('_')[0]
		if file_extension == ".pkl":
			pDF = pd.read_pickle(fh)
			if filename.startswith('merged_dataframe'):
				hashOfNormalizationTables['original'] = pDF
			else:
				hashOfNormalizationTables[pnom] = pDF
		elif file_extension == ".tsv":
			tDF = pd.read_csv(fh, sep='\t')
			hashOfNormalizationTables[pnom] = tDF
		else:
			raise ValueError('File Format not accounted for: '+fh)

	return hashOfNormalizationTables

if __name__ == "__main__":
	#norm_files = "${all_possible_tables}".split(' ')
	norm_files = "merged_dataframe_SET02_mod.pkl boxcox_transformed_SET02.tsv minmax_transformed_SET02.tsv quantile_transformed_SET02.tsv".split(' ')
	#myFileIdx = "${batchID}"	
	myFileIdx = "SET02"
	overrideVar = 'boxcox'

	## Make this a dictonary so it is expandable later, when adding more normalization approaches.
	allApproaches = buildDataDictionary(norm_files)
	
	pdfOUT = matplotlib.backends.backend_pdf.PdfPages("multinormalize_report_{}.pdf".format(myFileIdx))
	create_comparative_report(pdfOUT, myFileIdx, allApproaches)
	pdfOUT.close()
	
	if not overrideVar:
		print("No Override - Proceed to Auto Select")
		# Save the merged DataFrame as a pickle file
		allApproaches['original'].to_pickle('normalized_{}.pkl'.format(myFileIdx))
	else:
		print("Override Found")
		if overrideVar in allApproaches.keys():
			allApproaches[overrideVar].to_pickle('normalized_{}.pkl'.format(myFileIdx))
	
	
	
	
