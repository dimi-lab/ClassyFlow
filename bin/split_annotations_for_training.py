#!/usr/bin/env python3

import os, sys, csv, time
import pandas as pd
import fpdf
from fpdf import FPDF
import dataframe_image as dfi
from pprint import pprint

### Static Variables: File Formatting
classColumn = "${params.classifed_column_name}" 
batchColumn = 'Batch'
holdoutFraction = float("${params.holdout_fraction}") #0.05
cellTypeNegative = "${params.filter_out_junk_celltype_labels}".split(",")
cellTypeNegative.append("")
minimunHoldoutThreshold = ${params.minimum_label_count}


############################ PDF REPORTING ############################
def create_letterhead(pdf, WIDTH):
    pdf.image("${params.letterhead}", 0, 0, WIDTH)

def create_title(title, pdf):
    # Add main title
    pdf.set_font('Helvetica', 'b', 20)  
    pdf.ln(40)
    pdf.write(5, title)
    pdf.ln(10)
    # Add date of report
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(r=128,g=128,b=128)
    today = time.strftime("%d/%m/%Y")
    pdf.write(4, f'{today}')
    # Add line break
    pdf.ln(10)

def write_to_pdf(pdf, words):
    # Set text colour, font size, and font type
    pdf.set_text_color(r=0,g=0,b=0)
    pdf.set_font('Helvetica', '', 12)
    pdf.write(5, words)
############################ PDF REPORTING ############################



def stratified_sample(df, stratify_cols, frac=0.01, min_count=3):
    df = df.copy()
    df['OriginalIndex'] = df.index
    # Group by the stratification columns
    grouped = df.groupby(stratify_cols)
    # Function to sample 1% of the data or skip if less than min_count
    def sample_or_skip(group):
        if len(group) < min_count:
            return pd.DataFrame()  # Return an empty DataFrame if group has fewer rows than min_count
        return group.sample(frac=frac, random_state=42)

    # Apply the function to each group
    sampled_df = grouped.apply(sample_or_skip)
    #sampled_df = grouped.apply(sample_or_skip, include_groups=True)
    # Remove the multi-level index created by groupby
    sampled_df.reset_index(drop=True, inplace=True)
    return sampled_df


def gather_annotations(pickle_files):
    # Read the DataFrames from pickle files
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
    
    # Concatenate all DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    # merged_df = merged_df.sample(n=5000)  # remove this after testing
    merged_df[classColumn] = merged_df[classColumn].str.strip()
    merged_df = merged_df.dropna(subset=[classColumn])
    merged_df = merged_df.loc[~(merged_df[classColumn].isin(cellTypeNegative))]
    merged_df = merged_df.reset_index()

    ct = merged_df[classColumn].value_counts()
    pt = merged_df[classColumn].value_counts(normalize=True).mul(100).round(2).astype(str) + '%'
    
    # Get the stratified sample
    holdoutDF = stratified_sample(merged_df, [batchColumn, classColumn], frac=holdoutFraction, min_count=minimunHoldoutThreshold)
    print(f"holdoutDF {holdoutDF.shape}")
    #print(holdoutDF.columns.tolist())
    hd = holdoutDF[classColumn].value_counts()
    
    freqTable = pd.concat([ct,pt,hd], axis=1, keys=['counts', '%', 'holdout']).reset_index()
    freqTable.rename(columns={'index': classColumn}, inplace=True)
    #print(freqTable)
    # Apply styling to dataframe
    styled_df = freqTable.style.format({'Cell Types': "{}",
                      'Counts': "{:,}",
                      'Frequency': "{:.4f}%",
                      'Holdout': "{:,}"}).hide()
    dfi.export(styled_df, 'cell_count_table.png', max_rows=-1)

    keptFreq = freqTable[freqTable['holdout'].notna()]
    
    keptFreq.columns = keptFreq.columns.str.strip()
    #pprint(keptFreq)
    #print(keptFreq.columns.tolist())
    if classColumn not in keptFreq.columns:
        raise ValueError(f"{classColumn} column is missing. Found columns: {keptFreq.columns.tolist()}")
    ctl = keptFreq[classColumn].tolist()
    
    with open("celltypes.csv", 'w', newline='') as csvfile:
        f_writer = csv.writer(csvfile, delimiter=',')
        for ln in ctl:
            f_writer.writerow([ln])
    # holdoutDF = merged_df.groupby(batchColumn, group_keys=False).apply(lambda x: x.sample(frac=holdoutFraction))
    trainingDF = merged_df.loc[~merged_df.index.isin(holdoutDF['OriginalIndex'])]
    trainingDF = trainingDF[trainingDF[classColumn].isin(ctl)]
    trainingDF = trainingDF.reset_index(drop=True)
    print(f"trainingDF {trainingDF.shape}")
    print(trainingDF[classColumn].value_counts())
    
    # Remove the multi-level index created by groupby
    holdoutDF.reset_index(drop=True, inplace=True)
    holdoutDF.to_pickle('holdout_dataframe.pkl')
    trainingDF.to_pickle('training_dataframe.pkl')

if __name__ == "__main__":
    pickle_files = "${norms_pkl_collected}".split(' ')
    gather_annotations(pickle_files)
    
    # Create PDF
    # standard letter size is 215.9 by 279.4 mm or 8.5 by 11 inches.
    WIDTH = 215.9
    HEIGHT = 279.4
    pdf = FPDF() # A4 (210 by 297 mm)
    
    # Add 1st Page
    pdf.add_page()

    # Add lettterhead and title
    create_letterhead(pdf, WIDTH)
    create_title("Training Data Split", pdf)
    # Add some words to PDF
    write_to_pdf(pdf, "Holdout Fraction : {}".format(holdoutFraction))  
    pdf.ln(5)
    write_to_pdf(pdf, "Negative Class Value (to skip): {}".format(cellTypeNegative))
    pdf.ln(15)
    # Add table
    pdf.image('cell_count_table.png', w= (WIDTH*0.5) )
    # Generate the PDF
    pdf.output("annotation_report.pdf", 'F')




