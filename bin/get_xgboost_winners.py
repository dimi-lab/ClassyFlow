#!/usr/bin/env python3

import time
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
import dataframe_image as dfi
import pickle
from sklearn import preprocessing
import xgboost as xgb

############################ PDF REPORTING ############################
def create_letterhead(pdf, width, letterhead_path):
    pdf.image(letterhead_path, 0, 0, width)

def create_title(title, pdf):
    pdf.set_font('Helvetica', 'b', 20)
    pdf.ln(40)
    pdf.write(5, title)
    pdf.ln(10)
    pdf.set_font('Helvetica', '', 14)
    pdf.set_text_color(r=128, g=128, b=128)
    pdf.write(4, time.strftime("%d/%m/%Y"))
    pdf.ln(10)

def write_to_pdf(pdf, words):
    pdf.set_text_color(r=0, g=0, b=0)
    pdf.set_font('Helvetica', '', 12)
    pdf.write(5, words)
############################ PDF REPORTING ############################

def plot_parameter_search(df):
    df['combination'] = df.apply(lambda row: f"max_depth={row['max_depth']}, eta={row['eta']}", axis=1)
    mean_values = df.groupby('combination')['testf'].mean()
    max_comb = mean_values.idxmax()
    second_max_comb = mean_values.nlargest(2).idxmin()
    unique_combs = df['combination'].unique().tolist()
    color_palette = ['grey'] * len(unique_combs)
    color_palette[unique_combs.index(max_comb)] = 'red'
    color_palette[unique_combs.index(second_max_comb)] = 'orange'
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x='combination', y='testf', hue='combination', data=df,
        palette=color_palette, legend=False, flierprops={'markerfacecolor':'grey'}
    )
    plt.ylim(df['testf'].min() - 0.01, df['testf'].max() + 0.01)
    yticks = plt.gca().get_yticks()
    plt.gca().set_yticks(yticks)
    plt.gca().set_yticklabels(['{:.0f}%'.format(y * 100) for y in yticks])
    plt.xlabel('Combinations of Parameters')
    plt.ylabel('Test Accuracy')
    plt.title('Boxplot of XGB Training')
    plt.xticks(rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig("model_parameter_results.png", dpi=300, bbox_inches='tight')
    plt.close()

def make_a_new_model(toTrainDF, classColumn, cpu_jobs, mim_class_label_threshold, model_performance_table):
    class_counts = toTrainDF[classColumn].value_counts()
    print(class_counts)
    classes_to_keep = class_counts[class_counts > mim_class_label_threshold].index
    toTrainDF = toTrainDF[toTrainDF[classColumn].isin(classes_to_keep)]
    X = toTrainDF.select_dtypes(include=[np.number])
    X = X.loc[:, ~X.columns.duplicated()]
    le = preprocessing.LabelEncoder()
    y_Encode = le.fit_transform(toTrainDF[classColumn])
    unique, counts = np.unique(y_Encode, return_counts=True)
    plt.barh(unique, counts)
    plt.savefig("label_bars.png", dpi=300, bbox_inches='tight')
    np.save('classes.npy', le.classes_)
    num_round = 200

    xgboostParams = pd.read_csv(model_performance_table)
    plot_parameter_search(xgboostParams)
    xgboostParams['Training'] = xgboostParams['Training'].str.rstrip('%').astype(float) / 100
    summary_table = xgboostParams.groupby(['max_depth', 'eta']).agg(
        cv=('cv', lambda x: len(x.unique())),
        Training_mean=('Training', 'mean'),
        Training_std=('Training', 'std'),
        Test_mean=('testf', 'mean'),
        Test_std=('testf', 'std')
    ).reset_index()

    # --- Add this to handle parameter search being too small ---
    if len(summary_table) < 2:
        raise RuntimeError(
            f"Parameter search did not yield at least 2 unique parameter sets (found {len(summary_table)}). "
            "Multiple models cannot be compared. Please check your parameter search grid or input data."
        )
    # -----------------------------------------------------------

    styled_df = summary_table.style.format({
        'Training_mean': "{:,.2%}",
        'Training_std': "{:,.2%}",
        'Test_mean': "{:,.2%}",
        'Test_std': "{:,.2%}"
    })
    dfi.export(styled_df, 'parameter_search_results.png', table_conversion='matplotlib')

    # Train and save top 2 models
    top_models = summary_table.sort_values('Test_mean', ascending=False).drop_duplicates(subset=['max_depth', 'eta']).head(2)
    print("Top Models:", top_models)
    for fname, (_, row) in zip(["XGBoost_Model_First.pkl", "XGBoost_Model_Second.pkl"], top_models.iterrows()):
        param = {
            'max_depth': int(row['max_depth']),
            'eta': row['eta'],
            'objective': 'multi:softmax',
            'n_jobs': cpu_jobs,
            'num_class': len(unique),
            'eval_metric': 'mlogloss'
        }
        dtrainAll = xgb.DMatrix(X, label=y_Encode)
        bst = xgb.train(param, dtrainAll, num_round)
        pickle.dump(bst, open(fname, "wb"))

def main():
    parser = argparse.ArgumentParser(description="Train and select XGBoost models based on parameter search results.")
    parser.add_argument('--classColumn', required=True, help='Name of the classified column')
    parser.add_argument('--cpu_jobs', type=int, default=16, help='Number of CPU jobs to use')
    parser.add_argument('--mim_class_label_threshold', type=int, required=True, help='Minimum label count')
    parser.add_argument('--letterhead', required=True, help='Path to letterhead image')
    parser.add_argument('--model_performance_table', required=True, help='CSV with model performance')
    parser.add_argument('--trainingDataframe', required=True, help='Path to training dataframe pickle')
    parser.add_argument('--select_features_csv', required=True, help='Path to selected features CSV')
    args = parser.parse_args()

    myData = pd.read_pickle(args.trainingDataframe)
    with open(args.select_features_csv, 'r') as file:
        next(file)
        featureList = [line.strip() for line in file if line.strip()]
    if 'level_0' in featureList:
        featureList.remove('level_0')
    featureList.append(args.classColumn)
    focusData = myData[featureList]

    make_a_new_model(
        focusData,
        args.classColumn,
        args.cpu_jobs,
        args.mim_class_label_threshold,
        args.model_performance_table
    )

    WIDTH = 215.9
    pdf = FPDF()
    pdf.add_page()
    create_letterhead(pdf, WIDTH, args.letterhead)
    create_title("Model Training: XGBoost", pdf)
    write_to_pdf(pdf, "Selected Features: {}".format(', '.join(featureList)))
    pdf.ln(10)
    write_to_pdf(pdf, "Training Data {} cells by {} features".format(focusData.shape[0], focusData.shape[1]))
    pdf.ln(15)
    pdf.image('model_parameter_results.png', w=(WIDTH*0.95))
    pdf.ln(5)
    pdf.image('parameter_search_results.png', w=(WIDTH*0.4))
    pdf.output("Model_Development_Xgboost.pdf", 'F')

if __name__ == "__main__":
    main()

