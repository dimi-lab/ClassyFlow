#!/usr/bin/env python3

import os
import sys
import math
import re
import pandas as pd
import numpy as np
import argparse
import json
import logging

# Configure logging
logging.basicConfig(filename='add_empty_marker_noise.log',level=logging.INFO, format='[%(levelname)s] %(message)s')

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

def load_and_validate_panel_design(designFile):
    if not os.path.isfile(designFile):
        raise FileNotFoundError(f"Panel design file does not exist: {designFile}")
    panelDesign = pd.read_csv(designFile, index_col=0)
    if not (panelDesign.shape[0] > 0 and panelDesign.shape[1] > 0):
        raise ValueError("Panel design file is empty or malformed.")
    return panelDesign.clip(upper=1)

def resolve_batch_column(panelDesign, batchID_param):
    # 1. Try exact match first
    if batchID_param in panelDesign.columns:
        batch_col = batchID_param
        batch_prefix = batchID_param
        batch_suffix = ''
        logging.debug(f"Using exact batchID column: {batch_col}")
        return batch_col, batch_prefix, batch_suffix

    if '_fx' in batchID_param:
        batchID_param = batchID_param.replace('(_fx)$', '')
        logging.debug(f"Removed '_fx' from batchID, new value: {batchID_param}")

    # 2. Try to split batchID_param into prefix and suffix
    if '-' in batchID_param:
        batch_prefix = batchID_param.rsplit('-', 1)[0]
        batch_suffix = batchID_param.rsplit('-', 1)[1]
    else:
        batch_prefix = batchID_param
        batch_suffix = ''

 


    # 3. Find all columns that start with the prefix
    prefix_matches = [col for col in panelDesign.columns if col.startswith(batch_prefix)]
    if not prefix_matches:
        raise KeyError(f"No columns in panel design start with prefix '{batch_prefix}'. Available columns: {panelDesign.columns.tolist()}")

    # 4. If only one prefix match, use it
    if len(prefix_matches) == 1:
        batch_col = prefix_matches[0]
        logging.debug(f"Only one prefix match, using: {batch_col}")
        return batch_col, batch_prefix, batch_suffix

    # 5. Try to match both prefix and suffix
    suffix_matches = [col for col in prefix_matches if col.endswith(batch_suffix)] if batch_suffix else []
    if len(suffix_matches) == 1:
        batch_col = suffix_matches[0]
        logging.debug(f"Using prefix+suffix match: {batch_col}")
        return batch_col, batch_prefix, batch_suffix

    # 6. Try longest prefix match (most specific)
    longest_prefix = max(prefix_matches, key=len)
    longest_matches = [col for col in prefix_matches if len(col) == len(longest_prefix)]
    if len(longest_matches) == 1:
        batch_col = longest_matches[0]
        logging.debug(f"Using longest prefix match: {batch_col}")
        return batch_col, batch_prefix, batch_suffix

    # 7. If still ambiguous, log and raise error with more guidance
    logging.error(f"Ambiguous batch column for batchID '{batchID_param}'. Prefix matches: {prefix_matches}, Suffix matches: {suffix_matches if batch_suffix else 'N/A'}.")
    raise KeyError(
        f"Ambiguous or missing batch column for batchID '{batchID_param}'.\n"
        f"Prefix matches: {prefix_matches}\n"
        f"Suffix matches: {suffix_matches if batch_suffix else 'N/A'}\n"
        f"Tried exact, prefix, suffix, and longest prefix matching.\n"
        f"Consider making batchID or panel design columns more specific or unique."
    )

def get_missing_markers(panelDesign, batch_col, batch_prefix):
    prefix_batches = [col for col in panelDesign.columns if col.startswith(batch_prefix)]
    if not prefix_batches:
        raise KeyError(f"No columns in panel design start with prefix '{batch_prefix}' (checked in get_missing_markers). Available columns: {panelDesign.columns.tolist()}")
    logging.debug(f"Using prefix_batches for union: {prefix_batches}")
    logging.debug(f"panelDesign[{batch_col}] (first 10): {panelDesign[batch_col].head(10).tolist()}")
    mask = (panelDesign[batch_col] == 0) 
    debug_missing = panelDesign.index[mask].tolist()
    if not debug_missing:
        logging.info(f"No missing markers found for batch_col '{batch_col}' with prefix '{batch_prefix}'.")
    else:
        logging.info(f"Missing markers for batch_col '{batch_col}' with prefix '{batch_prefix}': {debug_missing}")
    return debug_missing

def check_header_conflicts(df, batchID_param):
    batchid_prefix = batchID_param
    header_matches = [col for col in df.columns if col.startswith(batchid_prefix)]
    if header_matches:
        raise ValueError(f"Header columns found that start with the batchID prefix '{batchid_prefix}': {header_matches}")

def generate_synthetic_columns(df, missingMarks, objtype):
    if not missingMarks:
        return df.copy(), 0

    new_columns = {}
    warning_count = 0

    for st in getUniqueSets(objtype):
        # Select columns matching the substring and force all to numeric
        commonSetFeatures = df.filter(regex=st).apply(pd.to_numeric, errors='coerce')
        logging.debug(f"'{st}'   => {str(commonSetFeatures.shape)}")

        if commonSetFeatures.shape[1] == 0:
            logging.warning(f"No columns found for set '{st}'. Skipping.")
            warning_count += 1
            continue
        
        # If too many warnings for CellObject, raise error
        if objtype == 'CellObject' and warning_count >= 5:
            raise ValueError("Too many missing sets for CellObject. Check if the correct object type is provided. At least 5 sets were missing.")
        
        descTbl = commonSetFeatures.describe([0.01,0.02,0.05,0.9])
        mn = descTbl.loc['min'].mean()
        mx = descTbl.loc['5%'].mean()

        filteredMissingFields = [f+st for f in missingMarks if f + st not in df.columns]
        if not filteredMissingFields:
            logging.warning(f"Skipping synthetic column '{col}' because it already exists in input DataFrame.")
            continue

        if pd.isna(mn) or pd.isna(mx):
            logging.warning(f"NaN detected for {st} (mn={mn}, mx={mx}), filling with zeros.")
            for col in filteredMissingFields:
                new_columns[col] = np.zeros(len(df))
        else:
            vals = np.random.uniform(low=mn, high=mx, size=(len(df), len(filteredMissingFields)))
            vals = np.clip(vals, 0, None)
            for i, col in enumerate(filteredMissingFields):
                new_columns[col] = vals[:, i]
        
    synthetic_features = len(new_columns)

    if new_columns:
        synthetic_df = pd.DataFrame(new_columns, index=df.index)
        prt1DataT = pd.concat([df, synthetic_df], axis=1, copy=False)
        logging.info(f"Added {synthetic_features} synthetic columns.")
        logging.debug(f"Headers after synthetic columns added: {list(prt1DataT.columns)}")
    else:
        prt1DataT = df.copy()

    return prt1DataT, synthetic_features

def write_output_files(prt1DataT, batchID_param, missingMarks, synthetic_features, original_features):
    expected_final_cols = original_features + synthetic_features
    if prt1DataT.shape[1] != expected_final_cols:
        raise ValueError(f"Final DataFrame shape mismatch: expected {expected_final_cols} columns, got {prt1DataT.shape[1]}")
    prt1DataT.to_pickle(f'merged_dataframe_{batchID_param}_mod.pkl')
    prt1DataT.to_csv(f'merged_dataframe_{batchID_param}_mod.csv', index=False)
    summary_report = {
        "batch_id": batchID_param,
        "original_features": original_features,
        "missing_markers": missingMarks,
        "synthetic_features_added": synthetic_features,
        "final_features": prt1DataT.shape[1],
        "percent_synthetic": (synthetic_features/prt1DataT.shape[1]*100) if prt1DataT.shape[1] > 0 else 0
    }
    with open(f'missing_data_fill_report_{batchID_param}.json', 'w') as f:
        json.dump(summary_report, f, indent=2)

def findMissingFeatures(df, batchID_param, prefix, designFile, objtype):
    panelDesign = load_and_validate_panel_design(designFile)
    batch_col, batch_prefix, batch_suffix = resolve_batch_column(panelDesign, prefix)
    missingMarks = get_missing_markers(panelDesign, batch_col, batch_prefix)
    check_header_conflicts(df, batchID_param)
    logging.info(f"For batch '{batchID_param}', missing markers to add: {missingMarks}")
    prt1DataT, synthetic_features = generate_synthetic_columns(df, missingMarks, objtype)
    if not missingMarks:
        logging.info("Skip this batch, no missing fields.")
        prt1DataT.to_pickle(f'merged_dataframe_{batchID_param}_mod.pkl')
        with open(f"missing_data_fill_report_{batchID_param}.json", "w") as f:
            json.dump({"skipped": True}, f)
        return
    write_output_files(prt1DataT, batchID_param, missingMarks, synthetic_features, df.shape[1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add noise columns for missing markers in quantification tables.")
    parser.add_argument('--objtype', required=True, help='QuPath object type (e.g., CellObject)')
    parser.add_argument('--pickleTable', required=True, help='Input pickle file with quantification table')
    parser.add_argument('--batchID', required=True, help='Batch ID for output file naming')
    parser.add_argument('--designTable', required=True, help='CSV file with panel design')

    args = parser.parse_args()


    # Assert input file existence
    if not os.path.isfile(args.pickleTable):
        raise FileNotFoundError(f"Input pickle file does not exist: {args.pickleTable}")
    myDataFile = pd.read_pickle(args.pickleTable)
    if myDataFile.empty:
        raise ValueError(f"Input pickle file is empty: {args.pickleTable}")

    # Vectorized NaN filling for numeric columns
    nan_cols = myDataFile.columns[myDataFile.isna().any()].tolist()
    if nan_cols:
        logging.warning("The following columns contain NaN values:")
        total_rows = len(myDataFile)
        for col in nan_cols:
            nan_count = myDataFile[col].isna().sum()
            percent = (nan_count / total_rows) * 100 if total_rows > 0 else 0
            logging.warning(f"  - {col}: {nan_count} NaN values ({percent:.2f}%)")

            # Skip if column is 100% NaN
            if nan_count == total_rows:
                logging.warning(f"    -> Skipping column '{col}' because it is 100% NaN.")
                continue

            # Vectorized check and fill for numeric columns
            non_nan_values = myDataFile[col].dropna()
            coerced = pd.to_numeric(non_nan_values, errors='coerce')
            if not coerced.isna().any():
                min_val = coerced.min()
                myDataFile[col] = pd.to_numeric(myDataFile[col], errors='coerce').fillna(min_val)
                logging.info(f"    -> Filled NaNs in numeric column '{col}' with minimum value: {min_val}")

    # Get batchID directly from the df:
    #Assert that this df only has 1 unique batchID
    assert myDataFile["original_batchID"].nunique(dropna=False) == 1
    prefix = str(myDataFile["original_batchID"].dropna().unique().item())

    myFileIdx = args.batchID
    panelCsvFile = args.designTable
    objtype = args.objtype

    findMissingFeatures(myDataFile, myFileIdx, prefix, panelCsvFile, objtype)



