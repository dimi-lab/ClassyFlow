#!/usr/bin/env python3

import os, sys, math
import pandas as pd
from sklearn.datasets import make_blobs
import argparse
import json

staticColHead = 'Unnamed: 0'

def getUniqueSets(objtype):
    uniqueSuffixes = []
    sts = ["Min","Max","Median","Mean","Std.Dev.","Variance"]
    if objtype == 'CellObject':
        comp = ["Nucleus","Cytoplasm","Membrane","Cell"]
        for c in comp:
            for s in sts:
                uniqueSuffixes.append(": "+c+": "+s)
    else:
        uniqueSuffixes = [": "+e for e in sts]
    return uniqueSuffixes

def findMissingFeatures(df, nom, designFile, objtype):
    panelDesign = pd.read_csv(designFile)
    pdf2 = panelDesign.loc[panelDesign[nom] == 0]
    print(pdf2)
    synthetic_features = 0
    if pdf2.shape[0] == 0:
        print("Skip this batch, no missing fields.")
        df.to_pickle('merged_dataframe_{}_mod.pkl'.format(nom))
        with open(f"missing_data_fill_report_{nom}.json", "w") as f:
            json.dump({"skipped": True}, f)
    else:
        missingMarks = pdf2[staticColHead].tolist()
        prt1DataT = df.copy(deep=True)
        for st in getUniqueSets(objtype):
            commonSetFeatures = df.filter(regex=st)
            print(f"'{st}'   => {str(commonSetFeatures.shape)}")
            print(commonSetFeatures.head())
            descTbl = commonSetFeatures.describe([0.01,0.02,0.05,0.9])
            descTbl['avg'] = descTbl.mean(axis=1)
            mn = descTbl.loc['min','avg']
            mx = descTbl.loc['5%','avg']
            center_box = (mn, mx)
            standard_dev = math.ceil((mx-mn)/6)

            theseMissingFields = [f+st for f in missingMarks]
            synthetic_features += len(theseMissingFields)
            if len(theseMissingFields) == 0:
                sys.exit('Missing Fields Empty! ( {} )'.format(st))
            vals, lbs = make_blobs(n_samples=len(df), n_features=len(theseMissingFields), center_box=center_box, cluster_std=standard_dev)
            dfTmp = pd.DataFrame(vals, columns=theseMissingFields)
            dfTmp[dfTmp < 0] = 0
            prt1DataT = pd.concat([prt1DataT,dfTmp],axis=1)

        prt1DataT.to_pickle('merged_dataframe_{}_mod.pkl'.format(nom))

        summary_report = {
            "batch_id": nom,
            "original_features": df.shape[1],
            "missing_markers": pdf2[staticColHead].tolist() if pdf2.shape[0] > 0 else [],
            "synthetic_features_added": synthetic_features if pdf2.shape[0] > 0 else 0,
            "final_features": prt1DataT.shape[1],
            "percent_synthetic": (synthetic_features/prt1DataT.shape[1]*100) if pdf2.shape[0] > 0 else 0
        }

        with open(f'missing_data_fill_report_{nom}.json', 'w') as f:
            json.dump(summary_report, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add noise columns for missing markers in quantification tables.")
    parser.add_argument('--objtype', required=True, help='QuPath object type (e.g., CellObject)')
    parser.add_argument('--bitDepth', required=True, help='Bit depth (not used in script, but required for compatibility)')
    parser.add_argument('--pickleTable', required=True, help='Input pickle file with quantification table')
    parser.add_argument('--batchID', required=True, help='Batch ID for output file naming')
    parser.add_argument('--designTable', required=True, help='CSV file with panel design')

    args = parser.parse_args()

    myDataFile = pd.read_pickle(args.pickleTable)
    if myDataFile.empty:
        sys.exit("Input pickle file is empty: {}".format(args.pickleTable))
    myFileIdx = args.batchID
    panelCsvFile = args.designTable
    objtype = args.objtype

    findMissingFeatures(myDataFile, myFileIdx, panelCsvFile, objtype)



