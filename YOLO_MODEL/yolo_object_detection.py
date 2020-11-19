import os
import cv2
import numpy as np
import shutil
from PIL import Image

print(cv2.__version__)

###
os.chdir("F:/Users/elect_09l/github/ship_detection/YOLO_MODEL")
###

dir_base = r"F:/Users/elect_09l/github/ship_detection"
dir_yolo = dir_base + "/YOLO_MODEL/"


dir_folder_list = [dir_yolo+"split_imgs", dir_yolo+"analysed_imgs", dir_yolo+"output"]
def delete_dir():
  for folder in dir_folder_list:
      try:
          if not os.path.exists(folder):
              continue
      except OSError:
          print ('Error: Deleting directory of: '+folder)
          
      shutil.rmtree(folder)

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
folder_list = ["split_imgs", "analysed_imgs", "output"]

def make_dir():
  for folder in folder_list:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print ('Error: Creating directory of data: '+folder)
''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
def create_images():
    cap = cv2.VideoCapture(dir_yolo + "input/" + "Patea_Bar_Crossing.mp4")

    currentFrame = 0

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if(ret == False):
            break
        # Saves image of the current frame in jpg file
        name = './split_imgs/frame' + str(currentFrame).zfill(5) + '.jpg'
        print ('Creating...' + name, end='\r')
        
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print("Created Images")

    if len(os.listdir(dir_yolo + "split_imgs") ) == 0:
        print("Directory is empty")


''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
# delete_dir()
# make_dir()
# create_images()

''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

# Load Yolo
#net = cv2.dnn.readNet(dir_yolo+"yolov3.weights", dir_yolo+"yolov3.cfg")
net = cv2.dnn.readNet("D:/py/project/YOLO/" + "yolov3.weights", dir_yolo+"yolov3.cfg")

classes = []
with open(dir_yolo+"coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image :  
# Image must be 16:9 aspect ratio !! maybe not actually?

dir_split_imgs = dir_yolo + "split_imgs"

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pathlib
contents = pathlib.Path("F:/Users/elect_09l/github/ship_detection/YOLO_MODEL/split_imgs").iterdir()

frame_number = 0
zeroes = 5

#### ~~ADDING EACH ANALYSED FRAME TO THE OUTPUT (VIDEO)~~ ####
CODEC = "MJPG"
assert len(CODEC)==4,"FOURCC code needs to have exactly four characters"
fourcc = cv2.VideoWriter_fourcc(CODEC[0],CODEC[1],CODEC[2],CODEC[3])

dimension_img = cv2.imread(dir_yolo+'split_imgs/frame00000.jpg')
if dimension_img is None:
  print("Could not load ", dir_yolo+'split_imgs/frame00000.jpg')
vw = dimension_img.shape[1] # use one of the images to determine width and height (in this case img 00000)
vh = dimension_img.shape[0] 
#1920, 1080



'''    '''
HD_VIDEO = True
'''    '''

if HD_VIDEO == True:
    vw = 1920
    vh = 1080 

print(vw)
print(vh)

fps = 25 # frame rate of output video
#writer = cv2.VideoWriter(dir_base+"demo.avi", fourcc, fps, (vw, vh), True) # original
writer = cv2.VideoWriter("F:/Users/elect_09l/github/ship_detection/YOLO_MODEL/output/"+"analysis_vid.avi", fourcc, fps, (vw, vh), True)

line_pts = []

for filename in os.listdir(dir_split_imgs):
    #img = cv2.imread(os.listdir(filename)) # original line

    # img = cv2.imread(dir_yolo+"/split_imgs/frame00000.jpg")
    #if (frame_number != 0):
    #    zeroes = 5-len(str(frame_number))

    # img = cv2.imread(dir_yolo+"split_imgs/frame"+str(zeroes)+str(frame_number)+".jpg")
    filled_number = str(frame_number).zfill(zeroes)
    f = dir_yolo+"split_imgs/frame"+str(filled_number)+".jpg"
    img = cv2.imread(f)
    
    assert img is not None, "Image not loaded "+f

    img = cv2.resize(img, (vw,vh), fx=0, fy=0) # resizes image to 1920 * 1080 pixels, Leave none if not setting resolution, fx and fy are scale factors
    
    height, width, channels = img.shape


    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    #print(indexes)

    #font = cv2.FONT_HERSHEY_DUPLEX

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            rect_colour = colors[class_ids[i]]
            rect_color = (255,0,0)  # Open CV uses BGR format
            pt_colour = (0,0,255)
            poly_colour = (0,255,0)

            
            if label == "boat": # Make sure that only detected boats are illustrated
                cv2.rectangle(img, (x, y), (x + w, y + h), rect_colour, 2) # Draws detection rectangle on detected object
                # cv2.putText(img, label, (x, y + 30), font, 3, rect_colour, 3)    # Text is not enabled
                
                pt_x = int(x+w/2)
                pt_y = int(y+h)
                pt_centre = (pt_x,pt_y) # centre of point to be drawn on image

                line_pts.append(pt_centre)

                pts_arr = np.array(line_pts)
                pts_arr = pts_arr.reshape((-1,1,2))
                
                for tup in line_pts:
                    np.append(pts_arr, tup)
                
                # print(len(line_pts))
                # print(len(pts_arr))

                cv2.circle(img, pt_centre, 5 ,pt_colour, -1)
                
                if (len(line_pts) > 1):  # Only draw the polygon after there are 2 points existing the the array
                    cv2.polylines(img, [pts_arr], False, poly_colour, 3)
                
    currentFrame = frame_number

    # Saves image of the current frame in jpg file
    name = './analysed_imgs/analysed_frame' + str(currentFrame).zfill(zeroes) + '.jpg'
    print ('Creating... ' + name) #, end='\r')
    
    cv2.imwrite(name, img)

    frame_number += 1

    # When everything done, release the capture
    
    cv2.destroyAllWindows()


    #cv2.imshow("Image", img)  # Display image in a window, can be changed to save to a file
    #cv2.waitKey(0)
    '''WRITE IMG TO VIDEO'''
    vid_img = img
    writer.write(img) 
    key = cv2.waitKey(3)#pauses for 3 seconds before fetching next image

    if key == 27: #if ESC is pressed, exit loop
        cv2.destroyAllWindows()
        break

print("Created Images ")

if len(os.listdir(dir_yolo + "analysed_imgs") ) == 0:
        print("Directory is empty")


cv2.destroyAllWindows()

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# print("line_pts:", line_pts)
# print("pts arr:", pts_arr)

writer.release() # put this at the end so that the file is closed

print("Released video ")
