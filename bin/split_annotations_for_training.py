#!/usr/bin/env python3

import os, sys, csv, time
import argparse
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


### Static Variables: File Formatting
batchColumn = 'Batch'

def stratified_sample(df, stratify_cols, frac=0.01, min_count=3):
    df = df.copy()
    df['OriginalIndex'] = df.index
    grouped = df.groupby(stratify_cols)
    def sample_or_skip(group):
        if len(group) < min_count:
            return group.iloc[0:0]  # empty frame with same columns
        return group.sample(frac=frac, random_state=42)
    sampled_df = grouped.apply(sample_or_skip)
    # Remove grouping columns added by groupby.apply (future-proof)
    if isinstance(sampled_df.index, pd.MultiIndex):
        sampled_df.reset_index(level=stratify_cols, drop=True, inplace=True)
    sampled_df.reset_index(drop=True, inplace=True)
    return sampled_df


def gather_annotations(pickle_files, classColumn, holdoutFraction, cellTypeNegative, minimunHoldoutThreshold):
    dataframes = []
    for file in pickle_files:
        print(f"Getting...{file}")
        if file.endswith('.pkl'):
            df = pd.read_pickle(file)
            dataframe_name = os.path.basename(file).replace('.pkl','').replace('merged_dataframe_','')
        else:
            df = pd.read_csv(file, sep='\t', low_memory=False)
            dataframe_name = os.path.basename(file).replace('.tsv','').replace('boxcox_transformed_','')
        df[batchColumn] = dataframe_name
        dataframes.append(df)
    
    merged_df = pd.concat(dataframes, ignore_index=True)
    merged_df[classColumn] = merged_df[classColumn].str.strip()
    merged_df = merged_df.dropna(subset=[classColumn])
    merged_df = merged_df.loc[~(merged_df[classColumn].isin(cellTypeNegative))]
    merged_df = merged_df.reset_index()

    ct = merged_df[classColumn].value_counts()
    pt = merged_df[classColumn].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
    
    holdoutDF = stratified_sample(merged_df, [batchColumn, classColumn], frac=holdoutFraction, min_count=minimunHoldoutThreshold)
    print(f"holdoutDF {holdoutDF.shape}")
    hd = holdoutDF[classColumn].value_counts()
    
    freqTable = pd.concat([ct,pt,hd], axis=1, keys=['counts', '%', 'holdout']).reset_index()
    freqTable.rename(columns={'index': classColumn}, inplace=True)
    # Export as HTML instead of PNG
    freqTable.to_html('cell_count_table.html', index=False)

    keptFreq = freqTable[freqTable['holdout'].notna()]
    keptFreq.columns = keptFreq.columns.str.strip()
    if classColumn not in keptFreq.columns:
        raise ValueError(f"{classColumn} column is missing. Found columns: {keptFreq.columns.tolist()}")
    ctl = keptFreq[classColumn].tolist()
    
    with open("celltypes.csv", 'w', newline='') as csvfile:
        f_writer = csv.writer(csvfile, delimiter=',')
        for ln in ctl:
            f_writer.writerow([ln])
    trainingDF = merged_df.loc[~merged_df.index.isin(holdoutDF['OriginalIndex'])]
    trainingDF = trainingDF[trainingDF[classColumn].isin(ctl)]
    trainingDF = trainingDF.reset_index(drop=True)
    print(f"trainingDF {trainingDF.shape}")
    print(trainingDF[classColumn].value_counts())
    
    holdoutDF.reset_index(drop=True, inplace=True)
    holdoutDF.to_pickle('holdout_dataframe.pkl')
    trainingDF.to_pickle('training_dataframe.pkl')

def create_html_report(letterhead_path, holdoutFraction, cellTypeNegative):
    from datetime import datetime
    html = []
    html.append('<!DOCTYPE html>')
    html.append('<html lang="en"><head>')
    html.append('<meta charset="UTF-8">')
    html.append('<title>Training Data Split Report</title>')
    html.append('<style>')
    html.append("""
        body { font-family: Helvetica, Arial, sans-serif; margin: 40px; }
        .letterhead { width: 100%; max-width: 900px; }
        .title { font-size: 2em; font-weight: bold; margin-top: 40px; }
        .subtitle { color: #888; font-size: 1.2em; margin-bottom: 20px; }
        .section { margin-top: 30px; }
        table { border-collapse: collapse; margin: 20px 0; }
        th, td { border: 1px solid #ccc; padding: 6px 12px; }
        th { background: #eee; }
    """)
    html.append('</style></head><body>')

    # Letterhead
    if letterhead_path and os.path.exists(letterhead_path):
        html.append(f'<img src="{letterhead_path}" class="letterhead" alt="Letterhead">')

    # Title and date
    html.append(f'<div class="title">Training Data Split</div>')
    html.append(f'<div class="subtitle">{datetime.now().strftime("%d/%m/%Y")}</div>')

    # Summary text
    html.append(f'<div class="section"><b>Holdout Fraction:</b> {holdoutFraction}</div>')
    html.append(f'<div class="section"><b>Negative Class Value (to skip):</b> {", ".join(cellTypeNegative)}</div>')

    # Table
    if os.path.exists('cell_count_table.html'):
        with open('cell_count_table.html') as f:
            table_html = f.read()
        html.append('<div class="section"><b>Cell Count Table</b><br>')
        html.append(table_html)
        html.append('</div>')

    html.append('</body></html>')

    with open("annotation_report.html", 'w') as f:
        f.write('\n'.join(html))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split annotations for training and holdout sets.")
    parser.add_argument('--classColumn', required=True, help='Name of the classified column')
    parser.add_argument('--holdoutFraction', type=float, required=True, help='Fraction of data to hold out')
    parser.add_argument('--cellTypeNegative', required=True, help='Comma-separated list of negative class values')
    parser.add_argument('--minimunHoldoutThreshold', type=int, required=True, help='Minimum label count for holdout')
    parser.add_argument('--pickle_files', required=True, help='Space-separated list of pickle files')
    parser.add_argument('--letterhead', required=True, help='Path to letterhead image for report')
    args = parser.parse_args()

    #print("Python version:")
    #print(sys.version)
    #print("Version info:")
    #print(sys.version_info)

    classColumn = args.classColumn
    holdoutFraction = args.holdoutFraction
    cellTypeNegative = args.cellTypeNegative.split(",")
    cellTypeNegative.append("")
    minimunHoldoutThreshold = args.minimunHoldoutThreshold
    pickle_files = args.pickle_files.split(' ')
    letterhead_path = args.letterhead

    gather_annotations(pickle_files, classColumn, holdoutFraction, cellTypeNegative, minimunHoldoutThreshold)
    create_html_report(letterhead_path, holdoutFraction, cellTypeNegative)




