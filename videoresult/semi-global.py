#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import os,shutil
from os.path import isfile, join
get_ipython().run_line_magic('matplotlib', 'inline')


# In[12]:


#creating video of left side view

pathIn= 'images/'
pathOut = 'left.avi'
fps = 17.0

frame_array = []
files = [f for f in os.listdir(pathIn) if (isfile(join(pathIn, f)) and f[1]=='1')]
 
#for sorting the file names properly
files.sort(key = lambda x: int(x[3:-4]))
 
for i in range(len(files)):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(filename)
    #inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()


#creating video of right side view

pathOut = 'right.avi'

frame_array = []
files = [f for f in os.listdir(pathIn) if (isfile(join(pathIn, f)) and f[1]=='2')]
 
#for sorting the file names properly
files.sort(key = lambda x: int(x[3:-4]))
 
for i in range(len(files)):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(filename)
    #inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()


# In[16]:


# lst=['bike']
#clearing answer folder
my_Folder_Name = 'answer' #This is a string that I generate
if not os.path.exists(my_Folder_Name):
    os.makedirs(my_Folder_Name)
folder=my_Folder_Name+'/'
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)
        
x=0
while True:
    count=0
    t = x
    while t>0:
        t=int(t/10)
        count+=1
    num=""
    for i in range(0,6-count):
        num = num + str(0)
    num = num + str(x)
    disparityrange = 160
    if x is 0 :
        num="000000"
    try:
        imgL=Image.open('images/I1_'+num+'.png')
        imgL=np.array(imgL)
        imgR=Image.open('images/I2_'+num+'.png')
        imgR=np.array(imgR)
        zero = np.zeros((imgL.shape[0],disparityrange))
    except:
        break
    imgL = np.hstack((zero,imgL))
    imgR = np.hstack((zero,imgR))
    
    imgL = imgL.astype(np.uint8)
    imgR = imgR.astype(np.uint8)
    
#     plt.figure(figsize=(10,10))
#     plt.imshow(imgL)
#     plt.title("Left")
#     plt.show()
    
#     plt.figure(figsize=(10,10))
#     plt.imshow(imgR)
#     plt.title("Right")
#     plt.show()
    
    window_size = 3  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    
    left_matcher = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=disparityrange,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=5,
        P1=8 * 3 * window_size ** 2,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size ** 2,
        disp12MaxDiff=-1,
        uniquenessRatio=15,
        speckleWindowSize=0,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    
    print('computing disparity...',num)
    
    displ = left_matcher.compute(imgL, imgR)#.astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)#.astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    #wls is used for smoothing images which are holefree
    
    #Disparity map filter based on Weighted Least Squares filter (in form of Fast Global Smoother that
    #is a lot faster than traditional Weighted Least Squares filter implementations)
    #and optional use of left-right-consistency-based confidence to refine the results in half-occlusions and uniform areas.
    lmbda = 80000
    sigma = 1.2
    visual_multiplier = 1

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    
    filteredImg = wls_filter.filter(displ, imgL, None, dispr)  # important to put "imgL" here!!!
    #When the normType is NORM_MINMAX, cv::normalize normalizes _src in such a way that
    #the min value of dst is alpha and max value of dst is beta. 
    #cv::normalize does its magic using only scales and shifts (i.e. adding constants and multiplying by constants).
    #dst = (x-a)/(b-a)*(beta-alpha) + alpha    a & b are lower and upper bound of normalization
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX);
    filteredImg = np.uint8(filteredImg)
    filteredImg=np.delete(filteredImg, np.s_[0:disparityrange], axis=1)
#     plt.figure(figsize=(20,10))
#     plt.imshow(filteredImg)
#     plt.title(num)
#     plt.show()
    path = 'answer/'
    # saving depth images in answer folder
    path = path + str(x) + '.jpg'
    cv2.imwrite(path, filteredImg)
    print(str(x) + ' done')
    x=x+1

pathIn= 'answer/'
pathOut = 'result.avi'

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]
 
#for sorting the file names properly
files.sort(key = lambda x: int(x[0:-4]))
 
for i in range(len(files)):
    filename=pathIn + files[i]
    #reading each files
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    print(filename)
    #inserting the frames into an image array
    frame_array.append(img)

out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
 
for i in range(len(frame_array)):
    # writing to a image array
    out.write(frame_array[i])
out.release()
   


# In[ ]:





# In[ ]:




