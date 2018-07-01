# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 14:43:23 2018

@author: akilj
"""

# importing requisite libraries
import torch
from torch.autograd import Variable
import cv2
from data import BaseTransform, VOC_CLASSES as labelmap
from ssd import build_ssd
import imageio

font = cv2.FONT_HERSHEY_SIMPLEX
posF = (20,50)
posS = (20,80)
fScale = 1
fColorG = (86, 239, 124)
fColorB = (98, 98, 249)
lineType = 2

# Defining the detect function to be applied to frames of video
def detect(frame, net, transform):
    '''
    Detect function based on SSD, frame by frame detection
    frame -> image
    net -> the ssd neural net
    transform -> transformation to be applied to img so compatible with the net
    '''
    height, width = frame.shape[:2]
    # transformed frame after appling transformations
    frame_t = transform(frame)[0] 
    # convert frame from numpy array into a torch tensor; higher dimensional matrix
    x = torch.from_numpy(frame_t)
    # have to switch order due to way ssd was trained
    x = x.permute(2, 0, 1)
    # in pytorch, we have to construct bathes to feed into the neural net and convert
    # into a Torch Variable
    x = Variable(x.unsqueeze(0))
    
    # Feed this torch variable into SSD neural net
    y = net(x)
    # Create new tensor which stores the values we are interested in
    detections = y.data
    scale = torch.Tensor([width, height, width, height])
    
    # detection contains [batch, num classes, num of occurences, (score, x0,y0,x1,y1)]
    # score is % accuracy of specifc classes found; now iterate through these vals
    for i in range(detections.size(1)):
        j = 0
        # only 1 batch
        while detections[0, i, j, 0] >= 0.6: # that value, index 0, is the score ^
            score = detections[0, i, j, 0]
            # values were normalized, so multiply by scale to restore to proper locations
            pt = (detections[0, i, j, 1:] * scale).numpy() # and convert to np array
            cv2.rectangle(frame, (int(pt[0]),int(pt[1])), (int(pt[2]), int(pt[3])), (255, 0, 0), 2)
            label = '{clabel}: {score}%'.format(clabel=labelmap[i-1],score=round(score*100,2))
            cv2.putText(frame,label,(int(pt[0]),int(pt[1])-20),font,fScale*2,fColorG,2, cv2.LINE_AA)
            j += 1
    return frame

# Creating the SSD neural network
net = build_ssd('test')
# load the weights for the neural nets
net.load_state_dict(torch.load('ssd300_mAP_77.43_v2.pth',
                               map_location= lambda storage, loc: storage)) # torch load opens the tensor

# Create the transform func
transform = BaseTransform(net.size,(104/256.0, 117/256.0, 123/256.0))

# Doing some Object Detection on a video
reader = imageio.get_reader('epic-horses.mp4')
fps = reader.get_meta_data()['fps']
writer = imageio.get_writer('labeled_video_horses.mp4',fps = fps)
for i, frame in enumerate(reader): # iterate through all frames of the reader
    frame = detect(frame,net.eval(),transform)
    writer.append_data(frame)
    print('Frame {} completed.'.format(i))
writer.close()



    
    
    