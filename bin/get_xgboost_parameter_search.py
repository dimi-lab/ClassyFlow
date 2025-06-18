#!/usr/bin/env python3

import argparse
import csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate XGBoost parameter search grid.")
    parser.add_argument('--max_cv', type=int, required=True, help='Maximum number of cross-validation iterations')
    parser.add_argument('--depth_start', type=int, default=2, help='Start of depth range (inclusive)')
    parser.add_argument('--depth_stop', type=int, default=22, help='End of depth range (exclusive)')
    parser.add_argument('--depth_step', type=int, default=4, help='Step size for depth range')
    parser.add_argument('--learnRates', type=str, default="0.1,0.7,1.0", help='Comma-separated list of learning rates')
    args = parser.parse_args()

    max_cv = args.max_cv
    depthFeild = range(args.depth_start, args.depth_stop, args.depth_step)
    learnRates = [float(x) for x in args.learnRates.split(",")]

    with open("xgb_iterate_params.csv", 'w', newline='') as csvfile:
        f_writer = csv.writer(csvfile)
        f_writer.writerow(["CVIDX", "DEPTH", "ETA"])
        for c in range(0, max_cv):
            for d in depthFeild:
                for l in learnRates:
                    f_writer.writerow([c, d, l])


