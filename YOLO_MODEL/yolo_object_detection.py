# Settings
HD_VIDEO = True  # Whether the output video is in native resolution or in 1920 * 1080p
VIDEO_NAME = "speedboat.mp4"  # Input video for analysis, must be in 'input' folder unless changed

BOOL_DELETE_DIR = True  # Delete any old images created from the program
BOOL_MAKE_DIR = True  # If BOOL_DELETE_DIR == True then leave as True to create the directories you just deleted
BOOL_CREATE_IMAGES = True  # Split the input image into the individual frames so they can be analysed, only turn off if the frames already exist
zeroes = 5  # Amount of zeroes to be filled for the frame numbers
BOOL_BOXES = True  # Do you want the detection boxes to be on?

import os
import cv2
import numpy as np
import shutil
from PIL import Image

os.chdir(
    "F:/Users/elect_09l/github/ship_detection/YOLO_MODEL"
)  # Set the base directory of the script and where other files will be created

dir_base = r"F:/Users/elect_09l/github/ship_detection"
dir_yolo = dir_base + "/YOLO_MODEL/"

dir_folder_list = [
    dir_yolo + "split_imgs",
    dir_yolo + "analysed_imgs",
    dir_yolo + "output",
]

# Delete the old directories so that all the old images and outputs are deleted, allowing new ones to be created
def delete_dir():
    for folder in dir_folder_list:
        try:
            if not os.path.exists(folder):
                continue
        except OSError:
            print("Error: Deleting directory of: " + folder)

        shutil.rmtree(folder)


folder_list = ["split_imgs", "analysed_imgs", "output"]

# Creating the directories for the frames and outputs to be stored in
def make_dir():
    for folder in folder_list:
        try:
            if not os.path.exists(folder):
                os.makedirs(folder)
        except OSError:
            print("Error: Creating directory of data: " + folder)


# Splitting the video into its individual frames
def create_images():
    cap = cv2.VideoCapture(dir_yolo + "input/" + VIDEO_NAME)

    currentFrame = 0

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == False:
            break
        # Saves image of the current frame in jpg file
        name = "./split_imgs/frame" + str(currentFrame).zfill(zeroes) + ".jpg"
        print("Creating..." + name, end="\r")

        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print("Created Images")

    if len(os.listdir(dir_yolo + "split_imgs")) == 0:
        print("Directory is empty")


if BOOL_DELETE_DIR:
    delete_dir()
if BOOL_MAKE_DIR:
    make_dir()
if BOOL_CREATE_IMAGES:
    create_images()


# Load YOLO Model
net = cv2.dnn.readNet(
    "D:/py/project/YOLO/" + "yolov3.weights", dir_yolo + "yolov3.cfg"
)  # load weights dir and configurations dir

classes = []
with open(dir_yolo + "coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image :
dir_split_imgs = dir_yolo + "split_imgs"

frame_number = 0

# ADDING EACH ANALYSED FRAME TO THE OUTPUT (VIDEO)
CODEC = "MJPG"
assert len(CODEC) == 4, "FOURCC code needs to have exactly four characters"
fourcc = cv2.VideoWriter_fourcc(CODEC[0], CODEC[1], CODEC[2], CODEC[3])

dimension_img = cv2.imread(dir_yolo + "split_imgs/frame00000.jpg")
if dimension_img is None:
    print("Could not load ", dir_yolo + "split_imgs/frame00000.jpg")

vw = dimension_img.shape[
    1
]  # Use one of the images to determine width and height (in this case img 00000)
vh = dimension_img.shape[0]


# Setting at beginning

if HD_VIDEO == True:
    vw = 1920
    vh = 1080

print(vw)
print(vh)

fps = 25  # Frame rate of output video

writer = cv2.VideoWriter(
    "F:/Users/elect_09l/github/ship_detection/YOLO_MODEL/output/" + "analysis_vid.avi",
    fourcc,
    fps,
    (vw, vh),
    True,
)

line_pts = (
    []
)  # List of all the points of detection by the model, tuples are added through each iteration of the loop below

first_iter = True

# Settings for the "Start Point" (shows the beginning of the path in which the ship moves in)
start_pt_radius = 15
start_pt_colour = (23, 144, 255)
start_pt_centre = (0, 0)

for filename in os.listdir(
    dir_split_imgs
):  # Iterates through every frame of the input video

    filled_number = str(frame_number).zfill(zeroes)
    f = dir_yolo + "split_imgs/frame" + str(filled_number) + ".jpg"
    img = cv2.imread(f)

    assert img is not None, "Image not loaded " + f

    img = cv2.resize(
        img, (vw, vh), fx=0, fy=0
    )  # Resizes image to 1920 * 1080 pixels, Leave none if not setting resolution, fx and fy are scale factors

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

    font = cv2.FONT_HERSHEY_DUPLEX
    pt_radius = 10

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            rect_colour = colors[class_ids[i]]
            rect_color = (255, 0, 0)  # Open CV uses BGR format
            pt_colour = (0, 0, 255)
            poly_colour = (0, 255, 0)

            if label == "boat":  # Make sure that only detected boats are illustrated
                if BOOL_BOXES:
                    cv2.rectangle(
                        img, (x, y), (x + w, y + h), rect_colour, 2
                    )  # Draws detection rectangle on detected object

                # cv2.putText(img, label, (x, y + 30), font, 3, rect_colour, 3)    # Text says what the image is (boat, person, etc.)

                pt_x = int(x + w / 2)
                pt_y = int(y + h)
                pt_centre = (pt_x, pt_y)  # centre of point to be drawn on image

                line_pts.append(pt_centre)

                pts_arr = np.array(line_pts)
                pts_arr = pts_arr.reshape((-1, 1, 2))

                for tup in line_pts:
                    np.append(pts_arr, tup)

                if first_iter:
                    start_pt_centre = pt_centre

                    start_pt_radius
                    first_iter = False

                pt_colour = (0, 0, 255)
                cv2.circle(
                    img, pt_centre, pt_radius, (0, 0, 255), -1
                )  # Circle dot to track boat

    cv2.circle(
        img, start_pt_centre, start_pt_radius, (0, 0, 255), -1
    )  # Dot to indicate starting point
    cv2.putText(
        img, "START", start_pt_centre, font, 3, (255, 0, 255), 3
    )  # Text says that this is the first point in the sequence

    if (
        len(line_pts) > 1
    ):  # Only draw the polygon after there are 2 points existing the the array
        cv2.polylines(img, [pts_arr], False, poly_colour, 3)

    currentFrame = frame_number

    # Saves image of the current frame in jpg file
    name = "./analysed_imgs/analysed_frame" + str(currentFrame).zfill(zeroes) + ".jpg"
    print("Creating... " + name, end="\r")  # )  """" ####

    cv2.imwrite(name, img)  # Write the analysed image to a file

    frame_number += 1

    cv2.destroyAllWindows()

    # WRITE IMG TO VIDEO
    vid_img = img
    writer.write(img)
    key = cv2.waitKey(3)  # Pauses for 3 seconds before fetching next image

    if key == 27:  # If ESC is pressed, exit loop
        cv2.destroyAllWindows()
        break

print("Created Images ")

if len(os.listdir(dir_yolo + "analysed_imgs")) == 0:
    print("Directory is empty")

cv2.destroyAllWindows()

writer.release()  # Put this at the end so that the file is closed

print("Released video ")