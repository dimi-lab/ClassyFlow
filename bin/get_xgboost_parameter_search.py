#!/usr/bin/env python3

import os, sys, csv

max_cv = int('${params.max_xgb_cv}')
depthFeild = range(2,22,4)
learnRates = [0.1,0.7,1.0]


with open("xgb_iterate_params.csv", 'w', newline='') as csvfile:
		f_writer = csv.writer(csvfile)
		f_writer.writerow(["CVIDX","DEPTH","ETA"])
		for c in range(0,max_cv):
			for d in depthFeild:
				for l in learnRates:
					f_writer.writerow([c, d, l])
			

