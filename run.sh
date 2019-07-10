#!/bin/bash

rm -rf ./src/tmp
matlab -nodesktop -nosplash -r ./src/ImgEnhance/PCNN_IE.m
python ./src/Salpredict/salpredict.py
matlab -nodesktop -nosplash -r ./src/ImgEnhance/Fusion.m

