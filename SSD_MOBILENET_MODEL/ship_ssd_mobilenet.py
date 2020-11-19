import os
import pathlib
import cv2

dir_models = "d:/apps/"

# Labels dir: D:/Apps/models/research/object_detection/data/mscoco_label_map.pbtxt
# Model Dir: C:\Users\elect_09l\.keras\datasets\ssd_mobilenet_v1_coco_2017_11_17\saved_model

''' ########################################################### '''
##
input_video = 'Patea_Bar_Crossing.mp4'
##
''' ########################################################### '''




"""### Imports"""

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

tf.config.list_physical_devices('GPU')

#tf.debugging.set_log_device_placement(True)

# Create some tensors
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)


###
os.chdir("F:/Users/elect_09l/github/ship_detection/SSD_MOBILENET_MODEL")
###

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from skimage.measure import block_reduce
"""Import the object detection module."""

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

import shutil
import glob

NR_IMAGES_TO_PROCESS = -1 # Set to -1 for all

base_path = "F:/Users/elect_09l/github/ship_detection/SSD_MOBILENET_MODEL/"

def delete_dir():
  dir_folder_list = [base_path+"data", base_path+"analysis_img", base_path+"proc_vid"]

  for folder in dir_folder_list:
      try:
          if not os.path.exists(folder):
              continue
      except OSError:
          print ('Error: Deleting directory of: '+folder)
          
      shutil.rmtree(folder)

folder_list = ["data", "analysis_img", "proc_vid"]

def make_dir():
  for folder in folder_list:
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
    except OSError:
        print ('Error: Creating directory of data: '+folder)


#######
'''
'''
#######

def create_images():
  # Playing video from file:
  vid_dir = base_path+"input/"
  cap = cv2.VideoCapture(vid_dir+input_video)

  currentFrame = 0

  while(True and (currentFrame<NR_IMAGES_TO_PROCESS or NR_IMAGES_TO_PROCESS<0)):
      # Capture frame-by-frame
      ret, frame = cap.read()
      if(ret == False):
          break
      # Saves image of the current frame in jpg file
      name = './data/frame' + str(currentFrame).zfill(5) + '.jpg'
      print ('Creating...' + name, end='\r')
      
      cv2.imwrite(name, frame)

      # To stop duplicate images
      currentFrame += 1

  # When everything done, release the capture
  cap.release()
  cv2.destroyAllWindows()
  print("Created Images")


delete_dir()
make_dir()
create_images()


"""Patches:"""

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile

"""# Model preparation

## Variables

Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing the path.

By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

## Loader
"""

def load_model(model_name):
  base_url = 'http://download.tensorflow.org/models/object_detection/'
  model_file = model_name + '.tar.gz'
  model_dir = tf.keras.utils.get_file(
    fname=model_name, 
    origin=base_url + model_file,
    untar=True)

  model_dir = pathlib.Path(model_dir)/"saved_model"

  print("Model dir", model_dir)

  model = tf.saved_model.load(str(model_dir))

  print("Loaded model from ",model_dir)

  return model

"""## Loading label map
Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine
"""

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = dir_models+'models/research/object_detection/data/mscoco_label_map.pbtxt'
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = pathlib.Path(dir_models+'models/research/object_detection/test_images/snapshot_serengeti')
dir_data = r"F:\Users\elect_09l\github\ship_detection\SSD_MOBILENET_MODEL\data"
# D:\py\project\vid_img_gen\data


PATH_TO_TEST_IMAGES_DIR = pathlib.Path(dir_data)
TEST_IMAGE_PATHS = sorted(list(PATH_TO_TEST_IMAGES_DIR.glob("*.jpg")))
TEST_IMAGE_PATHS

"""# Detection

Load an object detection model:
"""

model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
detection_model = load_model(model_name)

"""Check the model's input signature, it expects a batch of 3-color images of type uint8:"""

print(detection_model.signatures['serving_default'].inputs)

"""And returns several outputs:"""

detection_model.signatures['serving_default'].output_dtypes

detection_model.signatures['serving_default'].output_shapes

"""Add a wrapper function to call the model, and cleanup the outputs:"""

def run_inference_for_single_image(model, image):
  image = np.asarray(image)
  # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
  small_image = image # TODO block_reduce(image, block_size=(2,2,1), func=np.mean, cval=np.mean(image))
  input_tensor = tf.convert_to_tensor(small_image)
  # The model expects a batch of images, so add an axis with `tf.newaxis`.
  input_tensor = input_tensor[tf.newaxis,...]

  # Run inference
  model_fn = model.signatures['serving_default']
  output_dict = model_fn(input_tensor)  # THIS GOES WRONG

  # All outputs are batches tensors.
  # Convert to numpy arrays, and take index [0] to remove the batch dimension.
  # We're only interested in the first num_detections.
  num_detections = int(output_dict.pop('num_detections'))
  output_dict = {key:value[0, :num_detections].numpy() 
                 for key,value in output_dict.items()}
  output_dict['num_detections'] = num_detections

  # detection_classes should be ints.
  output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)
   
  # Handle models with masks:
  if 'detection_masks' in output_dict:
    # Reframe the the bbox mask to the image size.
    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
              output_dict['detection_masks'], output_dict['detection_boxes'],
               image.shape[0], image.shape[1])      
    detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5,
                                       tf.uint8)
    output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()
    
  return output_dict

"""Run it on each test image and show the results:"""

def show_inference(model, image_path):
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
  print("Open",image_path, end='\r')
  image_np = np.array(Image.open(image_path))
  assert image_np is not None, "Image did not load properly "+image_path
  assert len(image_np.shape)==3, "Image corrupted "+image_path
  assert image_np.shape[0]>100 and image_np.shape[1]>100, "Image size did not load properly "+image_path
  # Actual detection.
  output_dict = run_inference_for_single_image(model, image_np)
  # Visualization of the results of a detection.
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=8)

  #display(Image.fromarray(image_np))
  return image_np
s
'''
try:
    if not os.path.exists('analysis_img'):
        os.makedirs('analysis_img')
except OSError:
    print ('Error: Creating directory of data')
'''
print("Create video")
dir_base = "F:/Users/elect_09l/github/ship_detection/SSD_MOBILENET_MODEL/analysis_img/" # make sure to end with forward slash
CODEC = "MJPG"
assert len(CODEC)==4,"FOURCC code needs to have exactly four characters"
fourcc = cv2.VideoWriter_fourcc(CODEC[0],CODEC[1],CODEC[2],CODEC[3])

dimension_img = cv2.imread(base_path+'data/frame00000.jpg')
if dimension_img is None:
  print("Could not load ", base_path+'data/frame00000.jpg')
vw = dimension_img.shape[1] # use one of the images to determine width and height #320 
vh = dimension_img.shape[0] #240  

#1920, 1080

fps = 25 # frame rate of output video
#writer = cv2.VideoWriter(dir_base+"demo.avi", fourcc, fps, (vw, vh), True)
#writer = cv2.VideoWriter(dir_base+"demo.avi", fourcc, fps, (vw, vh), True)
writer = cv2.VideoWriter("F:/Users/elect_09l/github/ship_detection/SSD_MOBILENET_MODEL/proc_vid/"+"processed_vid.avi", fourcc, fps, (vw, vh), True)

current_frame = 0

print("Number of images",len(TEST_IMAGE_PATHS))

for image_path in TEST_IMAGE_PATHS[:NR_IMAGES_TO_PROCESS]:
  image_np = show_inference(detection_model, image_path)
  image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

  import random
  name = './analysis_img/proc_img' + str(current_frame).zfill(5) + '.jpg'
  #cv2.imshow("VIDEO_ANALYSIS_IMG", image_np)

  cv2.imwrite(name, image_np)
  assert image_np.shape[0:2] == (vh,vw), "Wrong image size for video"
  writer.write(image_np) # This is to convert from OpenCV format, otherwise just use writer.write(image)
  current_frame += 1

  key = cv2.waitKey(3)#pauses for 3 seconds before fetching next image

  if key == 27: #if ESC is pressed, exit loop
    cv2.destroyAllWindows()
    break
writer.release() # put this at the end so that the file is closed
print("Released video ")


clear_folders = False

if clear_folders:
  dir_folder_list = [base_path+"data", base_path+"analysis_img"]

  for folder in dir_folder_list:
      try:
          if not os.path.exists(folder):
              continue
      except OSError:
          print ('Error: Deleting directory of: '+folder)
          
      shutil.rmtree(folder)

