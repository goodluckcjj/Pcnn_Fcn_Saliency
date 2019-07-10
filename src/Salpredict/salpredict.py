import numpy as np
import sys
sys.path.append('../../caffe/python')
import caffe
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import os

root = '../../caffe/'
deploy = 'deploy.prototxt' 
caffe_model = '../model/train_iter_200000.caffemodel' 
imgpath = '../tmp'
imgs = os.listdir(imgpath)

caffe.set_mode_gpu()
caffe.set_device(0)
net = caffe.Net(deploy,caffe_model,caffe.TEST) 

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})  
transformer.set_transpose('data', (2,0,1)) 
transformer.set_raw_scale('data', 255)  
transformer.set_channel_swap('data', (2,1,0))  

if not os.path.exists(imgpath + 'result'):
    os.makedirs(imgpath + 'result')

print 'Starting coarse predictions:'
count = 1
for img in imgs:
    im = caffe.io.load_image(imgpath + img)    
    (H,W,C) = im.shape             
    net.blobs['data'].data[...] = transformer.preprocess('data', im)   
    out = net.forward()
    outmap = net.blobs['outmap'].data[0,0,:,:]
    map_final = cv2.resize(outmap,(W,H))
    map_final -= map_final.min()
    map_final /= map_final.max()
    map_final = np.ceil(map_final*255)
    name = imm.replace(".jpg", ".png")
    imgname = imgpath + 'result' + name
    cv2.imwrite(imgname, map_final)
    count += 1
    if count%100==0:
       print count,' saliency predictions is generated.'

print 'Done.'