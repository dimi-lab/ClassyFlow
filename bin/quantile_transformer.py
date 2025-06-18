#!/usr/bin/env python3

import sys, os, time
import fnmatch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.backends.backend_pdf
from sklearn.preprocessing import QuantileTransformer
import argparse

import fpdf
from fpdf import FPDF
import dataframe_image as dfi

############################ PDF REPORTING ############################
def create_title(title, pdf):
    pdf.set_font('Helvetica', 'b', 20)  
    pdf.ln(40)
    pdf.write(5, title)
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(r=128,g=128,b=128)
    today = time.strftime("%d/%m/%Y")
    pdf.write(4, f'{today}')
    pdf.ln(10)

def write_to_pdf(pdf, words):
    pdf.set_text_color(r=0,g=0,b=0)
    pdf.set_font('Helvetica', '', 12)
    pdf.write(5, words)
############################ PDF REPORTING ############################

def collect_and_transform(df, batchName, quantType, nucMark, plotFraction, quantileSplit):
    df['Image'] = [e.replace('.ome.tiff', '') for e in df['Image'].tolist() ]

    smTble = df.groupby('Slide').apply(lambda x: x.sample(frac=plotFraction)) 
    df_batching = smTble.filter(regex='(Mean|Median|Slide)',axis=1)
    df_melted = pd.melt(df_batching, id_vars=["Slide"])
    fig, ax1 = plt.subplots(figsize=(20,8))
    origVals = sns.boxplot(x='Slide', y='value', color="#CD7F32", data=df_melted, ax=ax1, showfliers = False)
    plt.setp(ax1.get_xticklabels(), rotation=40, ha="right")
    ax1.set_title('Combined Marker Distribution (original values)')
    fig = origVals.get_figure()
    fig.savefig("original_marker_sample_boxplots.png") 
    
    df_batching = smTble.filter(regex='(Mean|Median|Image|Slide)',axis=1)
    df_melted = pd.melt(df_batching, id_vars=["Image","Slide"])
    fig, ax1 = plt.subplots(figsize=(20,8))
    origVals = sns.boxplot(x='Image', y='value', hue="Slide", data=df_melted, ax=ax1, showfliers = False)
    plt.setp(ax1.get_xticklabels(), rotation=40, ha="right")
    ax1.set_title('Combined Marker Distribution (original values)')
    fig = origVals.get_figure()
    fig.savefig("original_marker_roi_boxplots.png") 
    
    if quantType == 'CellObject':
        df_batching2 = smTble.filter(regex='Cell: Mean',axis=1)
    else: 
        df_batching2 = smTble.filter(regex='Mean',axis=1)
        
    df_batching2 = df_batching2.loc[:, df_batching2.nunique() > 1]
    
    myFields = df_batching2.columns.to_list()
    NucOnly = list(filter(lambda x:nucMark in x, myFields))[0]
    for idx, fld in enumerate(myFields):
        if fld == NucOnly:
            continue
        denstPlt = df_batching2[[NucOnly,fld]].plot.density(figsize = (16, 8),linewidth = 3)
        plt.title("{} Distributions (original values)".format(fld))
        fig = denstPlt.get_figure()
        fig.savefig("original_value_density_{}.png".format(idx))
    
    scaler = QuantileTransformer(n_quantiles=quantileSplit, random_state=0)
    imgMets = df.filter(regex='(Min|Max|Median|Mean|StdDev)',axis=1)
    df_norm = pd.DataFrame(scaler.fit_transform(imgMets), columns=imgMets.columns)
    df_a = df[df.columns.difference(imgMets.columns)]
    bcDf = pd.concat([df_a.reset_index(drop=True), df_norm], axis=1).fillna(0)    

    smTble = bcDf.groupby('Slide').apply(lambda x: x.sample(frac=plotFraction)) 
    df_batching = smTble.filter(regex='(Mean|Median|Slide)',axis=1)
    df_melted = pd.melt(df_batching, id_vars=["Slide"])
    fig, ax1 = plt.subplots(figsize=(20,8))
    origVals = sns.boxplot(x='Slide', y='value', color="#50C878", data=df_melted, ax=ax1, showfliers = False)
    plt.setp(ax1.get_xticklabels(), rotation=40, ha="right")
    ax1.set_title('Combined Marker Distribution (quantile values)')
    fig = origVals.get_figure()
    fig.savefig("normlize_marker_sample_boxplots.png") 
    
    df_batching = smTble.filter(regex='(Mean|Median|Image|Slide)',axis=1)
    df_melted = pd.melt(df_batching, id_vars=["Image","Slide"])
    fig, ax1 = plt.subplots(figsize=(20,8))
    origVals = sns.boxplot(x='Image', y='value', hue="Slide", data=df_melted, ax=ax1, showfliers = False)
    plt.setp(ax1.get_xticklabels(), rotation=40, ha="right")
    ax1.set_title('Combined Marker Distribution (original values)')
    fig = origVals.get_figure()
    fig.savefig("normlize_marker_roi_boxplots.png") 
    
    colNames = list(filter(lambda x:'Mean' in x, df.columns.tolist()))
    NucOnly = list(filter(lambda x:nucMark in x, colNames))[0]
    for i in range(0, len(colNames), 4):
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        axs = axs.flatten()

        for j in range(4):
            if i + j < len(colNames):
                hd = colNames[(i + j)]
                nuc1 = pd.DataFrame({"Original_Value": df[NucOnly], "Transformed_Value":bcDf[NucOnly]})
                nuc1['Mark'] = nucMark
                mk2 = pd.DataFrame({"Original_Value": df[hd], "Transformed_Value":bcDf[hd]})
                mk2['Mark'] = hd.split(":")[0]
                qqDF = pd.concat([nuc1,mk2], ignore_index=True)

                ax2 = axs[j]
                sns.scatterplot(x='Original_Value', y='Transformed_Value', data=qqDF, hue="Mark", ax=ax2)
                ax2.set_title("Quantile Transformation: {}".format(hd))
                ax2.axline((0, 0), (nuc1['Original_Value'].max(), nuc1['Transformed_Value'].max()), linewidth=2, color='r')
            else:
                axs[j].axis('off')
        plt.tight_layout()
        fig.savefig("normlize_qrq_{}.png".format(i))
    bcDf.to_csv("quantile_transformed_{}.tsv".format(batchName), sep="\t")    

def generate_pdf_report(outfilename, batchName, letterhead):
    WIDTH = 215.9
    pdf = FPDF()
    pdf.add_page()
    create_title("Log Transformation: {}".format(batchName), pdf)
    pdf.image(letterhead, 0, 0, WIDTH)
    write_to_pdf(pdf, "Fig 1.a: Disrtibution of all markers combined summarized by biospecimen.")    
    pdf.ln(5)
    pdf.image('original_marker_sample_boxplots.png', w=WIDTH )
    pdf.ln(15)
    pdf.image('normlize_marker_sample_boxplots.png', w=WIDTH )
    pdf.ln(15)
    write_to_pdf(pdf, "Fig 1.b: Disrtibution of all markers combined summarized by images.")    
    pdf.ln(5)
    pdf.image('original_marker_roi_boxplots.png', w=WIDTH )
    pdf.ln(15)
    pdf.image('normlize_marker_roi_boxplots.png', w=WIDTH )
    pdf.ln(15)
    
    write_to_pdf(pdf, "Fig 2: Total cell population distibutions.")    
    for root, dirs, files in os.walk('.'):
        for file in fnmatch.filter(files, f"original_value_density_*"):
            pdf.image(file, w=WIDTH )
            pdf.ln(10)
            
    write_to_pdf(pdf, "Fig 3: Transformation Plots.")    
    for root, dirs, files in os.walk('.'):
        for file in fnmatch.filter(files, f"normlize_qrq_*"):
            pdf.image(file, w=WIDTH )
            pdf.ln(10)
    pdf.output(outfilename, 'F')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantile transform quantification tables and generate QC plots.")
    parser.add_argument('--pickleTable', required=True, help='Input pickle file')
    parser.add_argument('--batchID', required=True, help='Batch ID for output file naming')
    parser.add_argument('--quantType', required=True, help='QuPath object type (e.g., CellObject)')
    parser.add_argument('--nucMark', required=True, help='Nucleus marker name (e.g., DAPI)')
    parser.add_argument('--plotFraction', type=float, default=0.25, help='Fraction of data to plot for QC (default: 0.25)')
    parser.add_argument('--quantileSplit', type=int, required=True, help='Number of quantiles for QuantileTransformer')
    parser.add_argument('--letterhead', required=True, help='Path to letterhead image for PDF report')

    args = parser.parse_args()

    myData = pd.read_pickle(args.pickleTable)
    myFileIdx = args.batchID
    quantType = args.quantType
    nucMark = args.nucMark
    plotFraction = args.plotFraction
    quantileSplit = args.quantileSplit
    letterhead = args.letterhead

    collect_and_transform(myData, myFileIdx, quantType, nucMark, plotFraction, quantileSplit)
    generate_pdf_report("quantile_report_{}.pdf".format(myFileIdx), myFileIdx, letterhead)










