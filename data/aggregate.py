#!/usr/bin/python

import ezodf as od
import numpy as np
import time
import datetime
import os


data_root = "/mnt/data/Fer/diplomski/rad/data/"


# prebaci podatke o validation u results_agregated
res_val = data_root + "results_validation2.ods"

res_agg = data_root + "results_aggregated2.ods"



val_doc = od.opendoc(filename = res_val)

val_shs = val_doc.sheets

val_names = [s for s in val_shs.names()]
# get count of sheets
val_cnt = len(val_shs)

agg_doc = od.opendoc(filename = res_agg)

agg_shs = agg_doc.sheets

agg_all = agg_shs['all']


def aggregate(sheet, folder, epoch, data):
	
	# 2 - 301
	
	print("Received ", folder, epoch, data)
	
	for i in range(247, 247 + 1):#range(2, 301 + 1):
		
		print ('A' + str(i), 'B' + str(i))
		
		print(sheet['A' + str(i)].value, folder)
		print(sheet['B' + str(i)].value, float(epoch))
		if sheet['A' + str(i)].value == folder and str(int(sheet['B' + str(i)].value)) == epoch:
			
			print("Found ", i)
			
			for j in range(0, len(data)):
				
				sheet[chr(ord('H') + j) + str(i)].set_value(data[j])
			
			break
	
	


for i in range(0,1): #range(0, val_cnt):
	print(val_names[i])
	
	sheet = val_shs[i]
	
	if sheet['B2'].value == 'val' and sheet['C2'].value == 'reverberated' :
		
		data = [ cell.value for cell in [ sheet[chr(x) + "2"] for x in range(ord('D'), ord('J') + 1) ] ]
		
		tmp = sheet['A2'].value.strip()
		
		tmp = tmp.split('_')
		folder = tmp[0]
		epoch = tmp[1]
		
		print(folder, epoch, data)
		aggregate(agg_all, folder, epoch, data)
	else:
		print ("Table format different than expected")



agg_doc.save()

