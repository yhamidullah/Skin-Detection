import cv2
import numpy as np
from numpy import array
from matplotlib import pyplot as plt
from copy import deepcopy
import os, os.path
import pickle
from pathlib import Path
import time
from time import gmtime, strftime
import pandas as pd
import seaborn as sns
import sys
'''
0 > L > 100 ⇒ OpenCV range = L*255/100 (1 > L > 255)
-127 > a > 127 ⇒ OpenCV range = a + 128 (1 > a > 255)
-127 > b > 127 ⇒ OpenCV range = b + 128 (1 > b > 255)

'''
DATASET_DIR = "dataset"
INTERMEDIATE_DATAS_DIR = "intermediate_datas"
REPORTS_DIR = "reports"
REPORT_FILE = "reports.txt”"

SAVED_VALUES_DIR = "saved_values"

CONFIG_SAVED_FILE = "config.db"

CLASS_SKIN = 'SKIN'
CLASS_NON_SKIN = 'NON_SKIN'

SUFFIXE_SKIN = "_p"
SUFFIXE_NON_SKIN = "_np"

FILE_EXTENSION = ".jpg"


IMAGES_NUMBER = 0

HISTS = []


DATA_PIX= [0,0,0]

SCALE = 1

DIMENSION = int(256/SCALE)

CLASSES = [CLASS_SKIN, CLASS_NON_SKIN]
CLASS_SUFFIXES = [SUFFIXE_SKIN, SUFFIXE_NON_SKIN]

DIR_TRAIN = DATASET_DIR+'/train'
DIR_TEST = DATASET_DIR+'/test'

THRESHOLD_UPPER = 0.6
THRESHOLD_LOWER = 0.5

def train(CLASSES, EXTENSION=FILE_EXTENSION, SAVE_INTERMEDIATE_DATA=False, SHOW=False):
	#Welcome message
	print('-     - ------ -     ----   ---  - _ -  ----   -----  --- ')
	print(' - - -  |-- 	  -    |      |   | |   |  |--      |   |   |')
	print('  - -   -----  ----  ----   ---  -   -  ----     |    --- ')

	print(' -- --  -- --   ---- --  -- ' )
	print(' -----   --  --|      -  - ')
	print(' -- --   --     ----   --' )
	print('By ========>  HY(https://github.com/jassarpc)')
	print('__________________________________________________________')

	
	#Initilization of histograms
	for i in range(0,len(CLASSES)):
		# Filling 0 value to all
		HISTS.append([[0] * 256  for i in range(256)])
	IMAGES_NUMBER = len([name for name in os.listdir(DIR_TRAIN) if os.path.isfile(os.path.join(DIR_TRAIN, name))])
	IMAGES_NUMBER /=len(CLASSES)+1
	IMAGES_NUMBER = int(IMAGES_NUMBER)
	# Looping through images in dataset
	#print(IMAGES_NUMBER)
	print('****************')
	print('STARTED TRAIN :'+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	print('****************')
	for i in range(0,IMAGES_NUMBER):
		print('Image [',str(i+1),']..................................['+str(i+1)+'/'+str(IMAGES_NUMBER)+']')
		##### Original image
		orig = str(i+1)+EXTENSION
		img_orig = cv2.imread(DIR_TRAIN+'/'+orig)

		##### Conversion to CIELAB space
		#print(img_orig)
		img_orig_lab = cv2.cvtColor((img_orig/SCALE).astype(np.uint8), cv2.COLOR_BGR2LAB)

		##### Filename every class
		fnames = []
		for k in range(0, len(CLASSES)):
			fnames.append(str(i+1)+CLASS_SUFFIXES[k]+EXTENSION)

		##### Mask Images for every class
		masks = []
		for k in range(0, len(CLASSES)):
			masks.append(cv2.imread(DIR_TRAIN+'/'+fnames[k],0))
		
		##### Shape size for loop
		h = img_orig.shape[0]
		w = img_orig.shape[1]

		##### Looping through pixels
		for y in range(0, h-1):
			for x in range(0, w-1):
				# Temporary variables for A&B [We ignore the L]
				tmp_a = img_orig_lab[y,x][1]
				tmp_b = img_orig_lab[y,x][2]

				for k in range(0, len(CLASS_SUFFIXES)):
				# Increment count in HISTS[k] according to mask value
					if(masks[k][y,x] != 0):
						HISTS[k][tmp_a][tmp_b] +=1
						DATA_PIX[k]+=1
				DATA_PIX[2]+=1

		if(SAVE_INTERMEDIATE_DATA):
	  		for o in range(0,len(CLASSES)):
	  			cv2.imwrite(INTERMEDIATE_DATAS_DIR+'/'+str(i)+CLASS_SUFFIXES[o]+'_masked_by_'+CLASSES[o]+EXTENSION, cv2.bitwise_and(img_orig,img_orig,mask = masks[o]))
		  	if(not Path(SAVED_VALUES_DIR).is_dir()):
		  		os.mkdir(SAVED_VALUES_DIR)
		  	if(not Path(SAVED_VALUES_DIR+'/'+CONFIG_SAVED_FILE).is_file()):
		  		os.mknod(SAVED_VALUES_DIR+'/'+CONFIG_SAVED_FILE)
		  	f = open(SAVED_VALUES_DIR+'/'+CONFIG_SAVED_FILE, "wb")
		  	f.truncate(0)
		  	pickler = pickle.Pickler(f)
		  	pickler.dump([HISTS,DATA_PIX])
	print('****************')
	print('FINISHED TRAIN :'+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	print('****************')
	if SHOW:
		His = [[],[]]
		##### Histogramm normalization to get values between [0,1]
		for i in range(0,len(HISTS)):
			His[i].append([h / DATA_PIX[len(DATA_PIX)-1] for x in HISTS[i]])
		sns.distplot(His[0], hist=True, kde=False, bins=256);
		plt.xlim([0,256])
		plt.show()
	f = open(SAVED_VALUES_DIR+'/'+CONFIG_SAVED_FILE, "wb")
	f.truncate(0)
	pickler = pickle.Pickler(f)
	pickler.dump([HISTS,DATA_PIX])
	return HISTS

def test(ImagePath=DIR_TEST, EXTENSION=FILE_EXTENSION,SHOW=False):
	EVAL = array([0,0])
	IMAGES_NUMBER = len([name for name in os.listdir(DIR_TEST) if os.path.isfile(os.path.join(DIR_TEST, name))])
	IMAGES_NUMBER /=len(CLASSES)+1
	IMAGES_NUMBER = int(IMAGES_NUMBER)
	# Looping through images in dataset
	#print(IMAGES_NUMBER)
	print('****************')
	print('STARTED TEST :'+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	print('****************')
	for i in range(0,IMAGES_NUMBER):
		orig = str(i+1)+EXTENSION
		mask = str(i+1)+SUFFIXE_SKIN+EXTENSION
		EVAL += test_detect(ImagePath+'/'+orig,ImagePath+'/'+mask,i)
	print("-----------------------------")
	rate = round((EVAL[0]/(EVAL[0]+EVAL[1])),2)
	print("Correct : ",EVAL[0]," / ",EVAL[1])
	print("Correction rate : ",rate*100,"%")
	if(not Path(REPORTS_DIR+'/'+REPORT_FILE).is_file()):
		os.mknod(REPORTS_DIR+'/'+REPORT_FILE)
	file = open(REPORTS_DIR+'/'+REPORT_FILE,"w")
	file.write("******HY-VISION********"+"\n")
	file.write("******REPORT FILE*******"+"\n")
	file.write("**Date & time "+strftime("%Y-%m-%d %H:%M:%S", gmtime())+"***"+"\n")
	file.write("************************\n")
	file.write("Classes : "+str(len(CLASSES))+"\n")
	file.write("Image train : "+str(len([name for name in os.listdir(DIR_TRAIN) if os.path.isfile(os.path.join(DIR_TRAIN, name))]))+"\n")
	file.write("Image test : "+str(len([name for name in os.listdir(DIR_TEST) if os.path.isfile(os.path.join(DIR_TEST, name))]))+"\n")
	file.write("Pixel correct : "+str(EVAL[0])+" / "+str(EVAL[1])+"\n")
	file.write("Correction rate : "+str(rate*100)+"%\n")
	file.write("")
	file.write("Thank you for using HY-VISION\n")
	file.write("By HAMIDULLAH Yasser (yasserhamidullah@gmail.com) / httsp://github.com/Jassarpc\n")
	file.close() 
	print('****************')
	print('FINISHED TEST :'+strftime("%Y-%m-%d %H:%M:%S", gmtime()))
	print('****************')
def test_detect(filename,mask_name,c,SAVE=True):
	EVAL = array([0,0])
	img_orig = cv2.imread(filename)
	print(filename)
	img_orig_lab = cv2.cvtColor((img_orig/SCALE).astype(np.uint8), cv2.COLOR_BGR2LAB)
	img_mask = cv2.imread(mask_name, 0)
	h = img_orig.shape[0]
	w = img_orig.shape[1]
	mask = np.zeros((h,w,1), np.uint8)
	index = 0
	for y in range(0, h-1):
		for x in range(0, w-1):
			# Temporary variables for A&B [We ignore the L]
			tmp_a = img_orig_lab[y,x][1]
			tmp_b = img_orig_lab[y,x][2]
			start = time.clock()
			P  = calcProbabilities(HISTS,tmp_a,tmp_b)
			maxi = max(P)
			if(maxi[0]<=THRESHOLD_UPPER and CLASSES[P.index(maxi)]==CLASS_NON_SKIN):
				index = CLASSES.index(CLASS_SKIN)
			else:
				index = P.index(maxi)
			mask[y][x] = newPixel(index,)
			if mask[y][x] == img_mask[y][x]:
				EVAL[0]+=1
			else:
				EVAL[1]+=1	
	if SAVE:
		#cv2.imwrite('Input_image.jpg'+str(c),img_orig)
		kernel = np.ones((3,3),np.uint8)
		mask = cv2.dilate(mask,kernel,iterations = 1)
		#mask = cv2.erode(mask,kernel,iterations = 1)

		output_image = cv2.bitwise_and(img_orig,img_orig,mask = mask)
		kernel = np.ones((3,3),np.uint8)
		output_image = cv2.dilate(output_image,kernel,iterations = 1)
		output_image = cv2.erode(output_image,kernel,iterations = 1)
		#cv2.imshow('Output image',output_image)

		#Writing report files
		cv2.imwrite(REPORTS_DIR+'/test_input_'+filename,img_orig)
		cv2.imwrite(REPORTS_DIR+'/test_output_'+filename,output_image)
		cv2.imwrite(REPORTS_DIR+'/test_mask_'+filename,mask)
	rate = round(EVAL[0]/sum(EVAL), 2)*100
	print("-----------------------------")
	print("Correction rate : ",rate,"%")
	print("-----------------------------")
	return EVAL
def detect(filename,HIST=HISTS, PIXDATA=DATA_PIX, SAVE=True,SHOW=True):
	img_orig = cv2.imread(filename)
	print(filename)
	img_orig_lab = cv2.cvtColor(img_orig, cv2.COLOR_BGR2LAB)
	h = img_orig.shape[0]
	w = img_orig.shape[1]
	mask = np.zeros((h,w,1), np.uint8)
	index = 0
	for y in range(0, h-1):
		for x in range(0, w-1):
			# Temporary variables for A&B [We ignore the L]
			tmp_a = img_orig_lab[y,x][1]
			tmp_b = img_orig_lab[y,x][2]

			#Compute the probabilities
			P  = calcProbabilities(HIST,tmp_a,tmp_b,PIXDATA)

			#Taking the max probability through all 
			maxi = max(P)
			#Check if there is an NON_SKIN and less than THRESHOLD_UPPER, we take the SKIN_CLASSES
			if(maxi[0]<=THRESHOLD_UPPER and maxi[0]>=THRESHOLD_LOWER and CLASSES[P.index(maxi)]==CLASS_NON_SKIN):
				index = CLASSES.index(CLASS_SKIN)
			else:
				index = P.index(maxi)
			mask[y][x] = newPixel(index,)
			
	if SHOW:
		#Displaying the mask and input image
		cv2.imshow('mask',mask)
		cv2.imshow('Input image',img_orig)

		#Creating kernel for morphological transformations
		kernel = np.ones((3,3),np.uint8)

		#Applying the bitwise operation to get the output result
		output_image = cv2.bitwise_and(img_orig,img_orig,mask = mask)

		#Morphological transformations 
		output_image = cv2.dilate(output_image,kernel,iterations = 1)
		output_image = cv2.erode(output_image,kernel,iterations = 1)

		#Displaying the output image
		cv2.imshow('Output image',output_image)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
		if SAVE:
			#Writing images to file
			cv2.imwrite('detect_input_'+filename,img_orig)
			cv2.imwrite('detect_output_'+filename,output_image)
			cv2.imwrite('detect_mask_'+filename,mask)


def newPixel(index,C = CLASSES):
	if(C[index] == CLASS_SKIN):
		return "255"
	else:
		if(C[index] == CLASS_NON_SKIN):
			return "0"


def calcProbabilities(HIST, a,b, DATA_PIX=DATA_PIX):
	P_Classes = []
	P_fClasses = [] 
	for i in range(0,len(HIST)):
		P_Classes.append(DATA_PIX[i]/DATA_PIX[len(DATA_PIX)-1])
		P_fClasses.append(HIST[i][a][b])

	return bayes(P_Classes, P_fClasses)


def bayes(P_Classes, P_fClasses):
	P = []
	S = 0
	Sm = [a*b for a,b in zip(P_Classes,P_fClasses)]
	S = sum(Sm)
	if(S!=0):
		for i in range(0,len(P_Classes)):
			P.append([(P_Classes[i]*P_fClasses[i])/S])
	else:
		for i in range(0,len(P_Classes)):
			P.append([(P_Classes[i]*P_fClasses[i])/(S+1)])
	return P


def init():
	if len(sys.argv)>1:
		if not os.path.isfile(SAVED_VALUES_DIR+'/'+CONFIG_SAVED_FILE):
			HISTS = train(CLASSES,EXTENSION=FILE_EXTENSION, SAVE_INTERMEDIATE_DATA=True, SHOW=False)
			test()
			detect(str(sys.argv[1]))
		else:
			file = open(SAVED_VALUES_DIR+'/'+CONFIG_SAVED_FILE, 'rb')
			varse = pickle.load(file)
			HISTS = varse[0]
			DATA_PIX = varse[1]
			detect(str(sys.argv[1]), HISTS,DATA_PIX)
	else:
		print("Usage: python3 detection.py file_name or path")
		print(" ")
		print("Examples| python3 detection.py /home/username/folder/mimi.jpg")
		print("          python3 detection.py mimi.jpg")
init()
