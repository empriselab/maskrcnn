import os
import glob 

import cv2
import numpy as np

out = cv2.VideoWriter('ckpt_1.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v') , 20.0, (720,1280), isColor=True)

for filename in glob.glob('../data/interim/*.png'):
    img = cv2.imread(filename)
    out.write(img)

out.release()
