import csv
import glob
import numpy as np
import os
import pandas as pd
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog
import toml


def createmodel(modelfile):
    """
    Creates a kinematic model based on .toml file.

    :param modelfile: Pathway to .toml file containing model parameters.
    :return: Parameters of the model.
    """

    # Model settings
    model_toml = toml.load(modelfile)
    landmarks = model_toml['landmarks']['names']
    jointnames = model_toml['joints']['names']
    jointpts = model_toml['joints']['landmarks']
    segmentnames = model_toml['segments']['names']
    segmentpts = model_toml['segments']['landmarks']

    # Model checks
    if len(jointnames) != len(jointpts):
        print('Number of joint names and set of landmarks defining each joint is not the same.')
        quit()
    if len(segmentnames) != len(segmentpts):
        print('Number of segments and pair of landmarks defining each segment is not the same.')
        quit()

    model = KinematicModel(landmarks, jointnames, jointpts, segmentnames, segmentpts)

    return model


def createtrcfile(data, filename, markernames, fs):
    """
    Function to save raw trc file.

    Arguments:
        data: marker data formatted as (nframes x nmarkers x 3D).
        filename: path and name of file to be saved.
        markernames: list of marker names.
        fs: sampling frequency.
    """

    nframes = len(data)
    nmarkers = len(markernames)
    fs_str = '{:.2f}'.format(fs)

    rows = []
    rows.append(['PathFileType', '4', '(X/Y/Z)', filename])
    rows.append(['DataRate', 'CameraRate', 'NumFrames', 'NumMarkers', 'Units', 'OrigDataRate', 'OrigDataStartFrame',
                 'OrigNumFrames'])
    rows.append([fs_str, fs_str, str(nframes), str(nmarkers), 'mm', fs_str, '1', str(nframes)])
    rows.append(['Frame#', 'Time'])

    for marker in markernames:
        rows[3].append(marker)
        rows[3].extend(['', ''])

    rows.append(['', ''])

    for marker in range(1, nmarkers + 1):
        rows[4].extend(['X' + str(marker), 'Y' + str(marker), 'Z' + str(marker)])

    rows.append([])

    currentline = len(rows)
    for frame in range(nframes):
        rows.append([str(frame + 1), '{:.3f}'.format(frame / fs)])
        framedata = data[frame, :, :].reshape(-1)
        framedata_string = ['{:.3f}'.format(num) for num in framedata]
        rows[currentline].extend(framedata_string)
        currentline += 1

    # Saving as trc file
    with open(filename, 'w') as file:
        writer = csv.writer(file, delimiter='\t', lineterminator='\n')

        for row, content in enumerate(rows):
            writer.writerow(content)


def normalize(vector):
    """ Normalizes a vector to unit length 1.

    :param vector: Vector to be normalized.
    :return: Normalized vector.
    """
    vector_normalized = vector / np.linalg.norm(vector, axis=1)[:, None]

    return vector_normalized


class KinematicModel:
    def __init__(self, markers, jointname, jointinfo, segmentname, segmentinfo):
        self.markers = markers
        self.jointname = jointname
        self.jointinfo = jointinfo
        self.segmentname = segmentname
        self.segmentinfo = segmentinfo

    def markers(self):
        return self.markers()

    def jointname(self):
        return self.jointname()

    def jointinfo(self):
        return self.jointinfo()

    def segmentname(self):
        return self.segmentname()

    def segmentinfo(self):
        return self.segmentinfo()

    def nmarkers(self):
        return len(self.markers)

    def njoints(self):
        return len(self.jointname)

    def nsegments(self):
        return len(self.segmentname)


def calc_angles(model, data):
    """
    Calculates joint angles using the cosine formula.

    :param model: Model description.
    :param data: 3D marker data.
    :return: Joint angles.
    """

    # Create empty dataframe
    jointangles = pd.DataFrame(columns=model.jointname)

    # Run through each joint and calculate angles
    for joint in range(model.njoints()):
        points_id = [i for i in range(len(model.markers)) if model.markers[i] in model.jointinfo[joint]]
        points = data[:, points_id, :]
        v_prox = normalize(points[:, 0, :] - points[:, 1, :])
        v_dist = normalize(points[:, 2, :] - points[:, 1, :])
        angle = np.arccos(np.sum(v_prox * v_dist, axis=1))
        angle = 180 - np.rad2deg(angle)
        jointangles[model.jointname[joint]] = angle

    return jointangles


def calc_lengths(model, data):
    """
    Calculates segment lengths [mm].

    :param model: Model description.
    :param data: 3D marker data.
    :return: Segment lengths.
    """

    # Create empty dataframe
    lengths = pd.DataFrame(columns=model.segmentname)

    # Run through each joint and calculate angles
    for segment in range(model.nsegments()):
        points_id = [i for i in range(len(model.markers)) if model.markers[i] in model.segmentinfo[segment]]
        points = data[:, points_id, :]
        lengths[model.segmentname[segment]] = np.linalg.norm(points[:, 0, :] - points[:, 1, :], axis=1)

    return lengths


def calc_fingerlength(lengths):
    """
    Calculates finger lengths of the phalanges (not including metacarpals) [mm].

    :param lengths: Segment lengths.
    :return: Finger lengths.
    """

    # Create empty dataframe
    fingerlengths = pd.DataFrame(columns=['Thumb', 'Index', 'Middle', 'Ring', 'Little'])

    # Sum up segment lengths for each finger
    fingerlengths['Thumb'] = lengths['PP1'] + lengths['DP1']
    fingerlengths['Index'] = lengths['PP2'] + lengths['MP2'] + lengths['DP2']
    fingerlengths['Middle'] = lengths['PP3'] + lengths['MP3'] + lengths['DP3']
    fingerlengths['Ring'] = lengths['PP4'] + lengths['MP4'] + lengths['DP4']
    fingerlengths['Little'] = lengths['PP5'] + lengths['MP5'] + lengths['DP5']

    return fingerlengths


# Run code
if __name__ == '__main__':

    # Counter
    start = time.time()

    # Collection parameters
    fs = 30

    # Kinematic model
    Model = createmodel('model.toml')

    # Define working directory
    wdir = Path(os.getcwd())

    # Create a tkinter root window (it won't be displayed)
    root = tk.Tk()
    root.withdraw()

    # Open a dialog box to select participant's folder
    idfolder = filedialog.askdirectory(initialdir=str(wdir))
    id = os.path.basename(os.path.normpath(idfolder))

    # Grab all folders (3D landmarks from each camera combination)
    landmarksfolder = glob.glob(idfolder + '/landmarks/3d/*')

    # Create output folder
    outdir_kinematics = idfolder + '/kinematics/'
    if not os.path.exists(outdir_kinematics):
        os.mkdir(outdir_kinematics)

    # Running through each camera combination
    for folder in landmarksfolder:

        # Identify folder
        foldername = os.path.basename(folder)
        print(foldername)

        # Create output folder
        if not os.path.exists(outdir_kinematics + foldername):
            os.mkdir(outdir_kinematics + foldername)

        # Gather 3D hand locations from all trials
        trialdata = sorted(glob.glob(folder + '/*3Dlandmarks.npy'))

        # Calculates joint angles for each trial within each folder
        for trial in trialdata:

            # Identify trial name
            filename = os.path.basename(trial)
            fileparts = filename.split('_3Dlandmarks.npy')
            trialname = fileparts[0]
            # print(trialname)

            # Load 3D hand location data and reshape
            data_3d = np.load(trial)

            # Create .trc file for landmark locations
            # trcfilename = outdir_kinematics + trialname + '.trc'
            # createtrcfile(data_3d, trcfilename, Model.markers, fs)

            # Calculate angles and write to file
            angles = calc_angles(Model, data_3d)
            angles.to_csv(outdir_kinematics + foldername + '/' + trialname + '.csv')

            # Calculate segment and phalanx lengths
            # segmentlengths = calc_lengths(Model, data_3d)
            # phalanxlengths = calc_fingerlength(segmentlengths)
            # print(np.mean(phalanxlengths, axis=0))

    # Counter
    end = time.time()
    print('Time to run code: ' + str(end - start) + ' seconds')
