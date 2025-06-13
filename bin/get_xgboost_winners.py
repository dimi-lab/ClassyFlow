#!/usr/bin/env python3

import os, sys, re, random, math, time, glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint
import pickle

from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

import xgboost as xgb

#import fpdf
from fpdf import FPDF
import dataframe_image as dfi
from random import randint

## Static Variables: File Formatting
classColumn = "${params.classifed_column_name}" # 'Classification'
cpu_jobs=16
#leEncoderFile = "${params.output_dir}/models/classes.npy"
mim_class_label_threshold = ${params.minimum_label_count}


############################ PDF REPORTING ############################
def create_letterhead(pdf, WIDTH):
    pdf.image("${projectDir}/images/ClassyFlow_Letterhead.PNG", 0, 0, WIDTH)     
     
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


def plot_parameter_search(df):
    # Create the combinations of 'max_depth' and 'eta' for the x-axis
    df['combination'] = df.apply(lambda row: f"max_depth={row['max_depth']}, eta={row['eta']}", axis=1)
    # Calculate mean testf values for each combination
    mean_values = df.groupby('combination')['testf'].mean()

    # Find the combinations with the highest and second highest mean testf values
    max_mean_combination = mean_values.idxmax()
    second_max_mean_combination = mean_values.nlargest(2).idxmin()  # Get the second largest mean

    # Set the color palette
    unique_combinations = df['combination'].unique().tolist()
    color_palette = ['grey'] * len(unique_combinations)
    color_palette[unique_combinations.index(max_mean_combination)] = 'red'
    color_palette[unique_combinations.index(second_max_mean_combination)] = 'orange'

    # Plot the boxplot
    plt.figure(figsize=(12, 6))
    boxplot = sns.boxplot(x='combination', y='testf', data=df, palette=color_palette, flierprops={'markerfacecolor':'grey'})
    #boxplot = sns.boxplot(x='combination', y='testf', data=df, palette=color_palette, flierprops={'markerfacecolor':'grey'}, patch_artist=True)

    # Set y-axis limits and convert to percentages
    plt.ylim(df['testf'].min()-0.01, df['testf'].max()+0.01)
    y_labels = ['{:.0f}%'.format(y * 100) for y in plt.gca().get_yticks()]
    plt.gca().set_yticklabels(y_labels)

    # Add labels and title
    plt.xlabel('Combinations of Parameters')
    plt.ylabel('Test Accuracy')
    plt.title('Boxplot of XGB Training')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()

    # Show the plot
    plt.savefig("model_parameter_results.png", dpi=300, bbox_inches='tight')

def make_a_new_model(toTrainDF):
    allPDFText = {}
    class_counts = toTrainDF[classColumn].value_counts()
    print(class_counts)
    # Identify classes with fewer than 2 instances
    classes_to_keep = class_counts[class_counts > mim_class_label_threshold].index
    # Filter the dataframe to remove these classes
    toTrainDF = toTrainDF[toTrainDF[classColumn].isin(classes_to_keep)]
    X = toTrainDF[list(toTrainDF.select_dtypes(include=[np.number]).columns.values)]

    pprint(toTrainDF[classColumn])
    le = preprocessing.LabelEncoder()
    y_Encode = le.fit_transform(toTrainDF[classColumn])
    (unique, counts) = np.unique(y_Encode, return_counts=True)
    plt.barh(unique, counts)
    plt.savefig("label_bars.png", dpi=300, bbox_inches='tight')
    np.save('classes.npy', le.classes_)

    num_round = 200
    # specify parameters via map
    xgboostParams = pd.read_csv("${model_performance_table}")
    #xgboostParams = pd.read_csv("merged_xgb_performance_output.csv")
        
    #print(xgboostParams)
    plot_parameter_search(xgboostParams)
    
    # Group by 'max_depth' and 'eta' and aggregate
    xgboostParams['Training'] = xgboostParams['Training'].str.rstrip('%').astype(float) / 100
    summary_table = xgboostParams.groupby(['max_depth', 'eta']).agg(
        cv=pd.NamedAgg(column='cv', aggfunc=lambda x: len(x.unique())),
        Training_mean=pd.NamedAgg(column='Training', aggfunc='mean'),
        Training_std=pd.NamedAgg(column='Training', aggfunc='std'),
        Test_mean=pd.NamedAgg(column='testf', aggfunc='mean'),
        Test_std=pd.NamedAgg(column='testf', aggfunc='std')
    ).reset_index()
    styled_df = summary_table.style.format({'Max Tree Depth': "{}",
                      'ETA': "{:,}",
                      'CrossFold #': "{}",
                      'Training Acc. Mean': "{:,.2%}",
                      'Training Acc. Std.Dev.': "{:,.2%}",
                      'Test Acc. Mean': "{:,.2%}",
                      'Test Acc. Std.Dev.': "{:,.2%}"}).hide()
    dfi.export(styled_df, 'parameter_search_results.png', table_conversion='matplotlib')
    

    topIds = summary_table['Test_mean'].idxmax()
    rr = summary_table.iloc[topIds]
    param = {'max_depth':int(rr['max_depth']), 'eta': rr['eta'], 'objective':'multi:softmax', 'n_jobs': cpu_jobs,'num_class': len(unique), 'eval_metric': 'mlogloss' }
    dtrainAll = xgb.DMatrix(X, label=y_Encode)
    bst = xgb.train(param, dtrainAll, num_round)    
    # save model to file
    pickle.dump(bst, open("XGBoost_Model_First.pkl", "wb"))


    top2 = summary_table['Test_mean'].nlargest(2).idxmin()
    rr = summary_table.iloc[top2]
    param = {'max_depth':int(rr['max_depth']), 'eta': rr['eta'], 'objective':'multi:softmax', 'n_jobs': cpu_jobs,'num_class': len(unique), 'eval_metric': 'mlogloss' }
    dtrainAll = xgb.DMatrix(X, label=y_Encode)
    bst = xgb.train(param, dtrainAll, num_round)    
    # save model to file
    pickle.dump(bst, open("XGBoost_Model_Second.pkl", "wb"))



if __name__ == "__main__":
    myData = pd.read_pickle("${trainingDataframe}")
    with open("${select_features_csv}", 'r') as file:
        next(file) # Skip header
        featureList = file.readlines()
    featureList = list(set([line.strip() for line in featureList]))
    if 'level_0' in featureList:
        featureList.remove('level_0')
    featureList.append(classColumn)
    focusData = myData[featureList]   

    make_a_new_model(focusData) 
    
    WIDTH = 215.9
    HEIGHT = 279.4
    pdf = FPDF() # A4 (210 by 297 mm)
    pdf.add_page()
    # Add lettterhead and title
    create_letterhead(pdf, WIDTH)
    create_title("Model Training: XGBoost", pdf)
    # Add some words to PDF
    write_to_pdf(pdf, "Selected Features: {}".format(', '.join(featureList)))   
    pdf.ln(10)
    write_to_pdf(pdf, "Training Data {} cells by {} features".format(focusData.shape[0], focusData.shape[1]) )  
    pdf.ln(15)
    pdf.image('model_parameter_results.png', w= (WIDTH*0.95) )
    pdf.ln(5)
    pdf.image('parameter_search_results.png', w= (WIDTH*0.4) )
    # Generate the PDF
    pdf.output("Model_Development_Xgboost.pdf", 'F')


