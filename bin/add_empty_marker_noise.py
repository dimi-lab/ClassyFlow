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
    if '-' in nom:
        hyphen_index = nom.rfind('-')
        if hyphen_index != -1 and len(nom) - hyphen_index > 5:
            nom2 = nom[:hyphen_index]
        else:
            nom2 = nom
    else:
        nom2 = nom

    pdf2 = panelDesign.loc[panelDesign[nom2] == 0]
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
            #print(commonSetFeatures.head())

            # Only keep columns that are fully numeric after coercion
            numeric_cols = []
            for col in commonSetFeatures.columns:
                non_nan_values = commonSetFeatures[col].dropna()
                # Try to coerce all non-NaN values to numeric, and check for any NaN after coercion
                coerced = pd.to_numeric(non_nan_values, errors='coerce')
                # If any value could not be converted, skip this column
                if len(non_nan_values) > 0 and not coerced.isna().any():
                    numeric_cols.append(col)
                else:
                    print(f"[WARNING] Skipping non-numeric or mixed-type column '{col}' in set '{st}'.")
            if not numeric_cols:
                print(f"[WARNING] No numeric columns found for set '{st}'. Skipping.")
                continue

            commonSetFeatures = commonSetFeatures[numeric_cols].apply(pd.to_numeric, errors='coerce')
            descTbl = commonSetFeatures.describe([0.01,0.02,0.05,0.9])
            descTbl['avg'] = descTbl.mean(axis=1)

            if 'min' not in descTbl.index or '5%' not in descTbl.index:
                print(f"[WARNING] Could not find 'min' or '5%' in describe() for {st}. Skipping this set.")
                continue

            mn = descTbl.loc['min','avg']
            mx = descTbl.loc['5%','avg']

            theseMissingFields = [f+st for f in missingMarks]
            synthetic_features += len(theseMissingFields)
            if len(theseMissingFields) == 0:
                sys.exit('Missing Fields Empty! ( {} )'.format(st))

            # If mn or mx is NaN, fill with zeros instead of noise
            if pd.isna(mn) or pd.isna(mx):
                print(f"NaN detected for {st} (mn={mn}, mx={mx}), filling with zeros.")
                dfTmp = pd.DataFrame(0, index=df.index, columns=theseMissingFields)
            else:
                center_box = (mn, mx)
                standard_dev = math.ceil((mx-mn)/6)
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

    # Check for columns containing NaN values and print warning with percentage
    nan_cols = myDataFile.columns[myDataFile.isna().any()].tolist()
    if nan_cols:
        print("WARNING: The following columns contain NaN values:")
        total_rows = len(myDataFile)
        for col in nan_cols:
            nan_count = myDataFile[col].isna().sum()
            percent = (nan_count / total_rows) * 100 if total_rows > 0 else 0
            print(f"  - {col}: {nan_count} NaN values ({percent:.2f}%)")

            # Skip if column is 100% NaN
            if nan_count == total_rows:
                print(f"    -> Skipping column '{col}' because it is 100% NaN.")
                continue

            # Check if column is numeric (all non-NaN values are numbers)
            non_nan_values = myDataFile[col].dropna()
            coerced = pd.to_numeric(non_nan_values, errors='coerce')
            if not coerced.isna().any():
                # All non-NaN values are numeric
                min_val = coerced.min()
                myDataFile[col] = pd.to_numeric(myDataFile[col], errors='coerce').fillna(min_val)
                print(f"    -> Filled NaNs in numeric column '{col}' with minimum value: {min_val}")

    myFileIdx = args.batchID
    panelCsvFile = args.designTable
    objtype = args.objtype

    findMissingFeatures(myDataFile, myFileIdx, panelCsvFile, objtype)



