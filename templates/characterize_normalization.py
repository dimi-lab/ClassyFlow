#!/usr/bin/env python3

import sys, os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf

objtype = '${params.qupath_object_type}'
nucMark = '${params.nucleus_marker}'
downsampleFrac = ${params.downsample_normalization_plots}

def create_comparative_report(pdfOUT, nom, hshOfDFs):
	for ky, df in hshOfDFs.items():
		nucDF = df.filter(regex='('+nucMark+')')
		othDF = df[df.columns.drop(list(df.filter(regex='('+nucMark+')')))]
		## Subset to reduce compute burden.
		subNuc = nucDF.sample(frac=downsampleFrac)
		subOther = othDF.iloc[subNuc.index]
		
		if objtype == 'CellObject':
			# Create a new figure for each page
			fig, axs = plt.subplots(4, 2, figsize=(8.5, 11))
			axs = axs.flatten()
			
			c1 = list(filter(lambda x:'Cell: Mean' in x, nucDF.columns.tolist()))[0]
			subNuc[c1].plot.hist(bins=16,title='{} - {}'.format(ky,c1), ax=axs[0])
			
			c2 = list(filter(lambda x:'Cell: Mean' in x, subOther.columns.tolist()))
			tmp = subOther.melt(value_vars=c2, var_name='meteric', value_name='vals')
			tmp['vals'].plot.hist(bins=16,title='{} - {}'.format(ky,'All Other Cell: Mean'), ax=axs[1], color='g')
			
			sts = subOther[c2].mean(axis=0)
			d1 = sts.idxmin()
			d2 = sts.idxmax()
			subOther[d1].plot.hist(bins=16,title='{} Darkest - {}'.format(ky,d1), ax=axs[2], color='y')
			subOther[d2].plot.hist(bins=16,title='{} Brightest - {}'.format(ky,d2), ax=axs[3], color='k')
			
			c1 = list(filter(lambda x:'Nucleus: Mean' in x, nucDF.columns.tolist()))[0]
			subNuc[c1].plot.hist(bins=16,title='{} - {}'.format(ky,c1), ax=axs[4])
			
			c2 = list(filter(lambda x:'Nucleus: Mean' in x, subOther.columns.tolist()))
			tmp = subOther.melt(value_vars=c2, var_name='meteric', value_name='vals')
			tmp['vals'].plot.hist(bins=16,title='{} - {}'.format(ky,'All Other Nucleus: Mean'), ax=axs[5], color='g')

			sts = subOther[c2].mean(axis=0)
			d1 = sts.idxmin()
			d2 = sts.idxmax()
			subOther[d1].plot.hist(bins=16,title='{} Darkest - {}'.format(ky, d1), ax=axs[6], color='y')
			subOther[d2].plot.hist(bins=16,title='{} Brightest - {}'.format(ky, d2), ax=axs[7], color='k')

			# Adjust layout and save the page to the PDF
			plt.tight_layout()
			pdfOUT.savefig(fig)
			
		else:
			#raise ValueError('No code to assess other objects types in normalization comparison.')
			# Create a new figure for each page
			fig, axs = plt.subplots(2, 2, figsize=(8.5, 11))
			axs = axs.flatten()
			
			c1 = list(filter(lambda x:'Mean' in x, nucDF.columns.tolist()))[0]
			subNuc[c1].plot.hist(bins=16,title='{} - {}'.format(ky,c1), ax=axs[0])
			
			c2 = list(filter(lambda x:'Mean' in x, subOther.columns.tolist()))
			tmp = subOther.melt(value_vars=c2, var_name='meteric', value_name='vals')
			tmp['vals'].plot.hist(bins=16,title='{} - {}'.format(ky,'All Other Cell: Mean'), ax=axs[1], color='g')
			
			sts = subOther[c2].mean(axis=0)
			d1 = sts.idxmin()
			d2 = sts.idxmax()
			subOther[d1].plot.hist(bins=16,title='{} Darkest - {}'.format(ky,d1), ax=axs[2], color='y')
			subOther[d2].plot.hist(bins=16,title='{} Brightest - {}'.format(ky,d2), ax=axs[3], color='k')

			# Adjust layout and save the page to the PDF
			plt.tight_layout()
			pdfOUT.savefig(fig)
			
			


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
		# print([filename, file_extension])
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
	norm_files = "${all_possible_tables}".split(' ')
	myFileIdx = "${batchID}"
	overrideVar = "${params.override_normalization}"

	## Make this a dictonary so it is expandable later, when adding more normalization approaches.
	allApproaches = buildDataDictionary(norm_files)
	
	pdfOUT = matplotlib.backends.backend_pdf.PdfPages("multinormalize_report_{}.pdf".format(myFileIdx))
	create_comparative_report(pdfOUT, myFileIdx, allApproaches)
	pdfOUT.close()
	
	if not overrideVar:
		print("No Override - Proceed to Auto Select")
		# Save the merged DataFrame as a pickle file
		allApproaches['original'].to_pickle('normalized_{}.pkl'.format(myFileIdx))
		allApproaches['original'].to_csv("normalized_{}_{}.tsv".format('original',myFileIdx), sep="\t")
	else:
		print("Override Found")
		if overrideVar in allApproaches.keys():
			allApproaches[overrideVar].to_pickle('normalized_{}.pkl'.format(myFileIdx))
			allApproaches[overrideVar].to_csv("normalized_{}_{}.tsv".format(overrideVar,myFileIdx), sep="\t") 
		else:
			print(allApproaches.keys())
			sys.exit(1)
	
	
	
	
