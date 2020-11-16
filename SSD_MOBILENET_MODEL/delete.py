import os
import glob
import pandas as pd
import shutil

base_path = "F:/Users/elect_09l/github/ship_detection/SSD_MOBILENET_MODEL/"

dir_folder_list = [base_path+"data", base_path+"analysis_img", base_path+"proc_vid"]

for folder in dir_folder_list:
    try:
        if not os.path.exists(folder):
            continue
    except OSError:
        print ('Error: Deleting directory of: '+folder)
        
    shutil.rmtree(folder)
    
        


