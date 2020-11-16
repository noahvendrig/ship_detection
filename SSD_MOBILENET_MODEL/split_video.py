'''
Using OpenCV takes a mp4 video and produces a number of images.

Requirements
----
You require OpenCV 3.2 to be installed.

Run
----
Open the main.py and edit the path to the video. Then run:
$ python main.py

Which will produce a folder called data with the images. There will be 2000+ images for example.mp4.
'''
import cv2
import numpy as np
import os

folder_list = ["data", "analysis_img", "proc_vid"]
for folder in folder_list:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print ('Error: Creating directory of data: '+folder)



# Playing video from file:
cap = cv2.VideoCapture('example1.mp4')

try:
    if not os.path.exists('data'):
        os.makedirs('data')
except OSError as e:
    print ('Error: Creating directory of data')

currentFrame = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if(ret == False):
        break
    # Saves image of the current frame in jpg file
    name = './data/frame' + str(currentFrame).zfill(5) + '.jpg'
    print ('Creating...' + name)
    
    cv2.imwrite(name, frame)

    # To stop duplicate images
    currentFrame += 1

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()


#cap = cv2.VideoCapture('video.mp4')
