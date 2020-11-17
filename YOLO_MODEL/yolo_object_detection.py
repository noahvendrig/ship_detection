import os
import cv2
import numpy as np
import shutil

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


'''
def img_analysis(img):
    img = img
    #img = cv2.imread(dir_yolo+"/split_imgs/two-friends-sq.jpg")
    img = cv2.resize(img, None, fx=0.4, fy=0.4)
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
    print(indexes)
    #font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            #label = str(classes[class_ids[i]])
            #color = colors[class_ids[i]]
            color = 5000

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(img, label, (x, y + 30), font, 3, color, 3)    # Text is not enabled


    cv2.imwrite(dir_split_imgs, img )




    #cv2.imshow("Image", img)  # Display image in a window, can be changed to save to a file
    #cv2.waitKey(0)
'''

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import pathlib
contents = pathlib.Path("F:/Users/elect_09l/github/ship_detection/YOLO_MODEL/split_imgs").iterdir()

frame_number = 0
zeroes = 5

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



    img = cv2.resize(img, None, fx=0.4, fy=0.4)
    
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
    print(indexes)
    #font = cv2.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            #label = str(classes[class_ids[i]])
            #color = colors[class_ids[i]]
            color = 5000

            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            # cv2.putText(img, label, (x, y + 30), font, 3, color, 3)    # Text is not enabled


    #cv2.imwrite(dir_analysed_imgs, img )

    currentFrame = frame_number

    # Saves image of the current frame in jpg file
    name = './analysed_imgs/analysed_frame' + str(currentFrame).zfill(zeroes) + '.jpg'
    print ('Creating... ' + name, end='\r')
    
    cv2.imwrite(name, img)

    frame_number += 1

    # When everything done, release the capture
    
    cv2.destroyAllWindows()
    print("Created Images ")


    #cv2.imshow("Image", img)  # Display image in a window, can be changed to save to a file
    #cv2.waitKey(0)
    
if len(os.listdir(dir_yolo + "analysed_imgs") ) == 0:
        print("Directory is empty")


cv2.destroyAllWindows()


