#!/usr/bin/env python3

import os, sys, re, csv, time, warnings
import argparse
import pandas as pd
import numpy as np

# Static, internal pipeline fields
batchColumn = 'Batch'

def split_and_binarize(df, celltype, classColumn, varThreshold, mim_class_label_threshold, ifSubsetData, subSet_n):
    df["cnt"] = 1
    df["Lasso_Binary"] = 0
    df.loc[df[classColumn] == celltype, 'Lasso_Binary'] = 1

    # Skip if too few labels exist
    totCls = df["Lasso_Binary"].sum()
    print("{} has {} labels".format(celltype, totCls))
    if totCls < mim_class_label_threshold:
        print("{} '{}' is not enough class labels to model!".format(totCls, celltype))
        return

    # Optional parameter to speed up by reducing data amount. Half of target class size
    if ifSubsetData:
        totRow = df.shape[0]
        if totCls < subSet_n:
            df1 = df[df["Lasso_Binary"] == 1]
        else:
            df1 = df[df["Lasso_Binary"] == 1].sample(n=subSet_n)

        negN = totRow - totCls
        if negN < subSet_n:
            df2 = df[df["Lasso_Binary"] == 0]
        else:
            df2 = df[df["Lasso_Binary"] == 0].sample(n=subSet_n)

        df = pd.concat([df1, df2])

    # Remove all non-alphanumeric characters
    clean_celltype = re.sub(r'[^a-zA-Z0-9]', '', celltype)
    df.to_pickle('binary_df_{}.pkl'.format(clean_celltype))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split and binarize cell type labels for modeling.")
    parser.add_argument('--trainingDataframe', required=True, help='Path to the training dataframe pickle file')
    parser.add_argument('--celltype', required=True, help='Cell type label to binarize')
    parser.add_argument('--classColumn', default='Classification', help='Column name for cell type classification')
    parser.add_argument('--varThreshold', type=float, default=0.01, help='Variance threshold')
    parser.add_argument('--mim_class_label_threshold', type=int, required=True, help='Minimum class label threshold')
    parser.add_argument('--ifSubsetData', type=lambda x: (str(x).lower() == 'true'), default=True, help='Whether to subset data')
    parser.add_argument('--subSet_n', type=int, default=3000, help='Subset size for each class')

    args = parser.parse_args()

    myData = pd.read_pickle(args.trainingDataframe)
    myLabel = args.celltype.replace('[', '').replace(']', '')

    split_and_binarize(
        myData,
        myLabel,
        args.classColumn,
        args.varThreshold,
        args.mim_class_label_threshold,
        args.ifSubsetData,
        args.subSet_n
    )





