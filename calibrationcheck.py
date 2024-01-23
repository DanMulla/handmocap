# Libraries
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog
from labels2d import run_mediapipe
from triangulation import readcalibration
from triangulation import triangulate_simple
from triangulation import undistort_points
from triangulation import visualize_3d
from kinematics import createmodel
from kinematics import calc_angles
from kinematics import calc_fingerlength
from kinematics import calc_lengths


# Run code
if __name__ == '__main__':

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

    # Gather camera calibration parameters
    calfile = glob.glob(idfolder + '/calibration.toml')
    cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = readcalibration(calfile)
    ncams = len(cam_mats_intrinsic)

    # Collection parameters
    fs = 30

    # Kinematic model
    Model = createmodel('model.toml')

    # Gather trial folders
    trialfolders = sorted(glob.glob(idfolder + '/videos/*posturecheck*'))

    # Options for image and video saving (can increase code run time substantially)
    saveimages = False   # Takes up a lot of space, so caution if saving
    savevideo = False    # Can only be done if images saved

    for trial in trialfolders:

        # Identify trial name
        trialname = os.path.basename(trial)

        # Gather trial videos
        vidnames = sorted(glob.glob(trial + '/*.mp4'))
        ncams = len(vidnames)

        # Obtain 2D landmarks from each camera
        kpts_cam = run_mediapipe(vidnames, save_images=saveimages)
        data_2d = kpts_cam.reshape((ncams, -1, 2))

        # Undistort 2D points based on camera intrinsics and distortion coefficients
        # Output is ncams x (nframes x 21 landmarks) x 2-dimension
        data_2d_undistort = np.empty(data_2d.shape)
        for cam in range(ncams):
            data_2d_undistort[cam] = undistort_points(data_2d[cam].astype(float), cam_mats_intrinsic[cam],
                                                      cam_dist_coeffs[cam]).reshape(len(data_2d[cam]), 2)

        # Outputting 3D points
        npoints = data_2d_undistort.shape[1]  # nframes x 21
        data3d = np.empty((npoints, 3))
        data3d[:] = np.nan
        for point in range(npoints):
            subp = data_2d_undistort[:, point, :]
            data3d[point] = triangulate_simple(subp, cam_mats_extrinsic)

        # Reshaping to nframes x 21 lanmdarks x 3-dimension
        data3d = data3d.reshape((int(len(data3d)/21), 21, 3))

        # Output visualization
        visualize_3d(data3d)

        # Calculate angles
        angles = calc_angles(Model, data3d)
        angles['Time'] = np.arange(0, len(angles)/fs, 1/fs)
        angles['D2'] = angles['MCP2_Flexion'] + angles['PIP2_Flexion'] + angles['DIP2_Flexion']
        angles['D3'] = angles['MCP3_Flexion'] + angles['PIP3_Flexion'] + angles['DIP3_Flexion']
        angles['D4'] = angles['MCP4_Flexion'] + angles['PIP4_Flexion'] + angles['DIP4_Flexion']
        angles['D5'] = angles['MCP5_Flexion'] + angles['PIP5_Flexion'] + angles['DIP5_Flexion']

        # Calculate segment and phalanx lengths
        segmentlengths = calc_lengths(Model, data3d)
        phalanxlengths = calc_fingerlength(segmentlengths) / 10
        phalanxlengths['Time'] = np.arange(0, len(phalanxlengths)/fs, 1/fs)
        print('Mean Finger lengths (cm), should be around 6/8.5/9/8.5/7.2:')
        print(np.mean(phalanxlengths, axis=0))
        print('Std Finger lengths (cm):')
        print(np.std(phalanxlengths, axis=0))

        # Plot
        fig, axes = plt.subplots(nrows=2, ncols=1)
        angles.plot(ax=axes[0], x="Time", y=['D2', 'D3', 'D4', 'D5'])
        phalanxlengths.plot(ax=axes[1], x="Time", y=['Index', 'Middle', 'Ring', 'Little'])
        plt.show()

    # Counter
    end = time.time()
    print('Time to run code: ' + str(end - start) + ' seconds')
