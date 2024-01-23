# Libraries
from aniposelib.boards import CharucoBoard, Checkerboard
from aniposelib.cameras import Camera, CameraGroup
import glob
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog

# Code adapted from here: https://anipose.readthedocs.io/en/latest/aniposelib-tutorial.html

# Counter
start = time.time()

# Define working directory
wdir = Path(os.getcwd())

# Create a tkinter root window (it won't be displayed)
root = tk.Tk()
root.withdraw()

# Open a dialog box to select participant's folder
idfolder = filedialog.askdirectory(initialdir=str(wdir))
id = os.path.basename(os.path.normpath(idfolder))

# Gather calibration folder
calibrationfolder = glob.glob(idfolder + '/videos/calibration*')
if len(calibrationfolder) != 1:
    print('No or more than one set of calibration video folders exist.')
    quit()

# Gather calibration videos
vidnames = sorted(glob.glob(calibrationfolder[0] + '/*.mp4'))
vidnames = [[video] for video in vidnames]
ncams = len(vidnames)
if ncams == 0:
    print('No calibration videos found.')
    quit()
cam_names = [str(cam) for cam in range(ncams)]

# Board settings (square length in mm)
# board = Checkerboard(5, 7, square_length=31.9)
board = Checkerboard(5, 7, square_length=31.1)
# board = CharucoBoard(7, 10, square_length=25, marker_length=18.75, marker_bits=4, dict_size=50)

# Camera settings
cgroup = CameraGroup.from_names(cam_names, fisheye=False)

# Calibration (this can take several minutes depending on # of cameras and video duration)
# Board is first detected, then calibration is done based on detections using iterative bundle adjustment
cgroup.calibrate_videos(vidnames, board)

# Save calibration (intrinsic and extrinsic parameters)
cgroup.dump(idfolder + '/calibration.toml')

# Counter
end = time.time()
print('Time to run code: ' + str(end - start) + ' seconds')
