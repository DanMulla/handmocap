# Libraries
import cv2 as cv
import glob
from itertools import combinations
from labels2d import createvideo
import math
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
from scipy.interpolate import splev, splrep
from scipy import signal
import time
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import toml


def checksync():
    """
    Iterates through each camera pair, calculates cross-correlation and lag (if cross-correlation doesn't peak when
    no time shift present between signals).  If one camera is flagged across all pairs, then the data for that
    camera is shifted through next function (fixsync).
    """

    # Combination of all camera pairs
    camera_pairs = list(combinations(np.arange(0, ncams), 2))

    # Identifying task finger
    taskfinger = trialname.split('_')[2]
    fingers = ['index', 'middle', 'ring', 'little']
    tips = [8, 12, 16, 20]
    marker = tips[fingers.index(taskfinger)]

    # Pre-allocating empty arrays
    flag_pair = []  # List of camera pairs to be flagged
    flag_lag = []  # Magnitude of shift

    # Iterating through each camera pair and calculating cross-correlation between cameras
    for pair in camera_pairs:
        x = data_2d_right_spline[pair[0], :, marker, 0]
        y = data_2d_right_spline[pair[1], :, marker, 0]
        corr = signal.correlate(x, y, 'full')
        corr /= np.max(corr)
        lags = signal.correlation_lags(x.size, y.size, mode="full")
        lag = lags[np.argmax(corr)]

        # Flag if there is a lag (peak correlation is not at shift=0)
        if lag != 0:
            flag_pair.append(pair)
            flag_lag.append(lag)
            # corrcoef = corr[len(x) - 1:len(x)]

    # Evaluate flagged camera pairs
    if flag_pair:

        # Check if any flagged cameras appears across all pairs
        cam_flag, counts = np.unique(flag_pair, return_counts=True)
        cam_fix = cam_flag[counts == ncams - 1]

        # Shift data if there is detected lag across all camera pairs for the given camera
        if cam_fix.size > 0:
            fixsync(camlist=cam_fix, pairs_flagged=flag_pair, lags=flag_lag)


def fixsync(camlist, pairs_flagged, lags):
    """
    Shifts camera data as median value of lag between the given camera and all other cameras.
    """

    # Iterating through each camera flagged (works only if one camera shifted; need to adapt if > 1 camera)
    for cam in camlist:

        # Identify the pairs for the given camera and the associated lag (multiplying by -1 if first in pair)
        # Positive values indicate that given camera is slightly delayed (shifted forward)
        lag_sub = []
        for pair, lag in zip(pairs_flagged, lags):
            if cam in pair:
                if pair[0] == cam:
                    lag_sub.append(-1 * lag)
                else:
                    lag_sub.append(lag)

        # Calculating shift as median value (round down in case of even number)
        shift = math.floor(np.median(lag_sub))
        print('Shifting camera ' + str(cam) + ' by ' + str(shift) + ' frames.')

        # Shifting the data (adding first known or last known values to replace)
        data_2d_right_spline[cam, :, :, :] = np.roll(data_2d_right_spline[cam, :, :, :], shift, axis=0)
        if shift > 0:
            data_2d_right_spline[cam, :shift, :, :] = data_2d_right_spline[cam, shift, :, :]
        else:
            data_2d_right_spline[cam, shift:, :, :] = data_2d_right_spline[cam, shift, :, :]


def nan_helper(y):
    """
    https://github.com/lambdaloop/anipose/blob/master/anipose/filter_pose.py
    :param y:
    :return:
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def readcalibration(calibrationfile):
    """
    Outputs camera calibration parameters contained within a .toml file.

    :param calibrationfile: Calibration file pathway.
    :return: Extrinsic, intrinsic and distortion coefficients.
    """

    cal = toml.load(calibrationfile)
    ncams = len(cal) - 1
    extrinsics = []
    intrinsics = []
    dist_coeffs = []

    for cam in range(ncams):
        camname = 'cam_' + str(cam)

        # Camera extrinsic parameters
        cam_rotn = np.array(cal[camname]['rotation'])
        cam_transln = np.array(cal[camname]['translation'])
        cam_transform = transformationmatrix(cam_rotn, cam_transln)
        extrinsics.append(cam_transform)

        # Camera intrinsic parameters
        cam_int = np.array(cal[camname]['matrix'])
        intrinsics.append(cam_int)

        # Camera distortion coefficients
        cam_dist = np.array(cal[camname]['distortions'])
        dist_coeffs.append(cam_dist)

    return extrinsics, intrinsics, dist_coeffs


def rotationmatrix(r):
    """
    Create rotation matrix from a rotation vector.

    :param r: Axis of rotation.
    :return: 3x3 rotation matrix.
    """

    theta = np.linalg.norm(r)
    if theta == 0:
        return np.eye(3)
    else:
        axis = r / theta
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)
        return R


def smooth2d(data2d, kernel, pixeldif):
    """
    Applies a median filter to the data and calculates difference between original and filtered signal.
    If difference between signals is beyond certain threshold, a spline is fit to the original data.

    Adapted from here:
    https://github.com/lambdaloop/anipose/blob/master/anipose/filter_pose.py
    :param data2d: 2D data.
    :param kernel: kernel size for median filter.
    :param pixeldif: Threshold length of pixels difference to warrant spline smooth.
    :return: Spline fit 2D data.
    """

    # Empty array for storing median filtered signal
    data_2d_mfilt = np.empty(data2d.shape)

    # Applying median filter
    for camera in range(ncams):
        for landmark in range(21):
            data_2d_mfilt[camera, :, landmark, 0] = signal.medfilt(data2d[camera, :, landmark, 0], kernel_size=kernel)
            data_2d_mfilt[camera, :, landmark, 1] = signal.medfilt(data2d[camera, :, landmark, 1], kernel_size=kernel)

    # Calculating difference between original and filtered signal
    errx = data2d[:, :, :, 0] - data_2d_mfilt[:, :, :, 0]
    erry = data2d[:, :, :, 1] - data_2d_mfilt[:, :, :, 1]
    err = np.sqrt((errx ** 2) + (erry ** 2))

    # Applying spline fit to replace extraneous data points
    data_2d_spline = np.empty(data2d.shape)

    for camera in range(ncams):
        for landmark in range(21):
            x = data2d[camera, :, landmark, 0]
            y = data2d[camera, :, landmark, 1]
            err_sub = err[camera, :, landmark]
            bad = np.zeros(err_sub.shape, dtype='bool')
            bad[err_sub >= pixeldif] = True

            # Ignore first and last few data points
            bad[:kernel] = False
            bad[-kernel:] = False

            pos = np.array([x, y]).T
            posi = np.copy(pos)
            posi[bad] = np.nan

            for i in range(posi.shape[1]):
                vals = posi[:, i]
                nans, ix = nan_helper(vals)

                # More than 1 data point missing, more than 80% data there
                if np.sum(nans) > 0 and np.mean(~nans) > 0.80:
                    spline = splrep(ix(~nans), vals[~nans], k=3, s=0)
                    vals[nans] = splev(ix(nans), spline)

                data_2d_spline[camera, :, landmark, i] = vals

    return data_2d_spline


def switch_hands(righthand, lefthand, axis=0):
    """
    Finds instances where the right and left hands may have been mis-identified by mediapipe,
    and switches the hands.
    2D data of right and left hand as inputs (# cams x # frames x # landmarks x 2D)

    For our setup:
    (1) The right hand should always be to the left of the left hand.  This is flagged
    by the indices identified in the "switched" variable.
    (2) The right hand should always be on screen.  This is flagged by the indices identified
    in the "missing" variable.

    Note: this is only a temporary solution for our setup and is not a general fix for other setups.
    """

    righthand_copy = righthand.copy()
    lefthand_copy = lefthand.copy()

    for cam in range(righthand.shape[0]):
        switched = np.where(np.logical_and(righthand[cam, :, 0, axis] > lefthand[cam, :, 0, axis],
                                           lefthand[cam, :, 0, 0] != -1))
        missing = np.where(np.logical_and(righthand[cam, :, 0, 0] == -1,
                                          lefthand[cam, :, 0, 0] != -1))
        replace = np.concatenate(switched + missing)
        if replace.size > 0:
            righthand[cam, replace, :, :] = lefthand_copy[cam, replace, :, :]
            lefthand[cam, replace, :, :] = righthand_copy[cam, replace, :, :]

    return righthand, lefthand


def transformationmatrix(r, t):
    """
    Create a 4x4 transformation matrix based on a rotation vector and translation vector.

    :param r: 3x3 rotation matrix.
    :param t: translation vector.
    :return: 4x4 transformation matrix.
    """

    R = rotationmatrix(r)
    T = np.concatenate((R, t.reshape(3, 1)), axis=1)
    T = np.vstack((T, [0, 0, 0, 1]))
    return T


def triangulate_simple(points, camera_mats):
    """
    Triangulates undistorted 2D landmark locations from each camera to a set of 3D points in global space.

    Code from here: https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/cameras.py

    :param points: 2D camear landmark locations.
    :param camera_mats: Camera extrinsic matrices.
    :return: 3D points.
    """

    num_cams = len(camera_mats)
    A = np.zeros((num_cams * 2, 4))
    for i in range(num_cams):
        x, y = points[i]
        mat = camera_mats[i]
        A[(i * 2):(i * 2 + 1)] = x * mat[2] - mat[0]
        A[(i * 2 + 1):(i * 2 + 2)] = y * mat[2] - mat[1]
    u, s, vh = np.linalg.svd(A, full_matrices=True)
    p3d = vh[-1]
    p3d = p3d[:3] / p3d[3]
    return p3d


def undistort_points(points, matrix, dist):
    """
    Undistorts 2D pixel points based on camera intrinsics and distortion coefficients.

    Code from here: https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/cameras.py

    :param points: 2D pixel points of landmark locations.
    :param matrix: Intrinsic camera parameters.
    :param dist: Distortion coefficients.
    :return: Undistorted 2D points of landmark locations.
    """

    points = points.reshape(-1, 1, 2)
    out = cv.undistortPoints(points, matrix, dist)
    return out


def hex2bgr(hexcode):
    """
    Converts hexadecimal code to BGR (OpenCV reverses the RGB).
    Adapted from here: https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python

    :param hexcode: Hexadecimal code
    :return: BGR tuple
    """
    h = hexcode.lstrip('#')
    rgb = tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))
    bgr = rgb[::-1]

    return bgr


def visualizelabels(input_streams, data):
    """
    Draws 2D hand landmarks on videos.

    :param input_streams: List of videos
    :param data: 2D hand landmarks.
    """

    # Create a list of cameras based on input_streams
    caps = [cv.VideoCapture(stream) for stream in input_streams]

    # Set camera resolution
    for cap in caps:
        width = int(cap.get(3))
        height = int(cap.get(4))
        cap.set(3, height)
        cap.set(4, width)

    # Creating links for each digit
    colors = ['#DDDDDD', '#EE3377', '#EE7733', '#009988', '#0077BB']
    colors = [item for item in colors for _ in range(4)]

    links = [[0, 1], [1, 2], [2, 3], [3, 4],
             [0, 5], [5, 6], [6, 7], [7, 8],
             [0, 9], [9, 10], [10, 11], [11, 12],
             [0, 13], [13, 14], [14, 15], [15, 16],
             [0, 17], [17, 18], [18, 19], [19, 20]]

    # Initialize frame number
    framenum = 0

    while True:

        # Read frames from videos
        frames = [cap.read() for cap in caps]

        # If wasn't able to read, break
        if not all(ret for ret, _ in frames):
            break

        # Convert frames from BGR to RGB
        for cam, (_, frame) in enumerate(frames):
            frames[cam] = (True, cv.cvtColor(frame, cv.COLOR_BGR2RGB))

        # To improve performance, optionally mark the image as not writeable to pass by reference
        for cam, (_, frame) in enumerate(frames):
            frames[cam] = (True, frame.copy())
            frame.flags.writeable = False

        # Access 2D hand landmarks (pixel coordinates) if detected (otherwise [-1, -1])
        for cam, (ret, frame) in enumerate(frames):

            # Draw hand landmarks
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            for number, link in enumerate(links):
                start = link[0]
                end = link[1]
                if np.isnan(data[cam, framenum, start, 0]) or np.isnan(data[cam, framenum, end, 0]):
                    continue
                posn_start = (int(data[cam, framenum, start, 0]), int(data[cam, framenum, start, 1]))
                posn_end = (int(data[cam, framenum, end, 0]), int(data[cam, framenum, end, 1]))
                cv.line(frame, posn_start, posn_end, hex2bgr(colors[number]), 2)

            for landmark in range(21):
                if np.isnan(data[cam, framenum, landmark, 0]):
                    continue
                posn = (int(data[cam, framenum, landmark, 0]), int(data[cam, framenum, landmark, 1]))
                cv.circle(frame, posn, 3, (0, 0, 0), thickness=1)

            # Display and save images
            cv.imshow(f'cam{cam}', frame)
            cv.imwrite(outdir_images_refined + trialname + '/cam' + str(cam) + '/' + 'frame' + f'{framenum:04d}' + '.png',
                       frame)

        k = cv.waitKey(10)
        if k & 0xFF == 27:  # ESC key
            break

        # Increment frame number
        framenum += 1

    # Clear windows
    cv.destroyAllWindows()
    for cap in caps:
        cap.release()


def visualize_3d(p3ds, save_path=None):
    """
    Visualized 3D points in 3D space and saves images if filename given.

    Code adapted from here: https://github.com/TemugeB/bodypose3d/blob/main/show_3d_pose.py

    :param p3ds: 3D points
    :param save_path: Filename of saved images.
    """

    # Creating links for each digit
    thumb = [[0, 1], [1, 2], [2, 3], [3, 4]]
    index = [[0, 5], [5, 6], [6, 7], [7, 8]]
    middle = [[0, 9], [9, 10], [10, 11], [11, 12]]
    ring = [[0, 13], [13, 14], [14, 15], [15, 16]]
    little = [[0, 17], [17, 18], [18, 19], [19, 20]]
    body = [thumb, index, middle, ring, little]
    colors = ['#AAAAAA', '#EE3377', '#EE7733', '#009988', '#0077BB']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Determine axis ranges (ignoring first and last second)
    axis_min = np.min(p3ds[30:-30], axis=(0, 1))
    axis_max = np.max(p3ds[30:-30], axis=(0, 1))
    axisrange = axis_max - axis_min
    max_axisrange = max(axisrange)
    max_axisrange = (math.ceil(max_axisrange / 100.00) * 100)

    for framenum, kpts3d in enumerate(p3ds):

        # Skip frames
        # if framenum % 3 == 0:
        #     continue

        # Drawing links
        for bodypart, part_color in zip(body, colors):
            for _c in bodypart:
                ax.plot(xs=[kpts3d[_c[0], 0], kpts3d[_c[1], 0]], ys=[kpts3d[_c[0], 1], kpts3d[_c[1], 1]],
                        zs=[kpts3d[_c[0], 2], kpts3d[_c[1], 2]], linewidth=5, c=part_color, alpha=0.7)

        # Drawing joints
        for i in range(21):
            ax.scatter(xs=kpts3d[i:i + 1, 0], ys=kpts3d[i:i + 1, 1], zs=kpts3d[i:i + 1, 2],
                       marker='o', s=40, lw=2, c='white', edgecolors='black', alpha=0.7)

        # Axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        # Axis limits
        ax.set_xlim3d([axis_min[0], axis_min[0]+max_axisrange])
        ax.set_xlabel('X')
        ax.set_ylim3d([axis_min[1], axis_min[1]+max_axisrange])
        ax.set_ylabel('Y')
        ax.set_zlim3d([axis_min[2], axis_min[2]+max_axisrange])
        ax.set_zlabel('Z')
        ax.view_init(-71, -73)

        # Remove background
        ax.set_axis_off()

        if save_path is not None:
            plt.savefig(save_path.format(framenum), dpi=100)
        else:
            plt.pause(0.1)
        ax.cla()

    if save_path is None:
        plt.show()

    plt.close(fig)


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
    id = os.path.split(os.path.split(idfolder)[0])[1]
    visit = os.path.basename(os.path.normpath(idfolder))
    print(id + '; ' + visit)

    # Gather camera calibration parameters
    calfile = glob.glob(idfolder + '/calibration.toml')
    cam_mats_extrinsic, cam_mats_intrinsic, cam_dist_coeffs = readcalibration(calfile)
    ncams = len(cam_mats_intrinsic)

    # Gather all combination of cameras
    camlist = np.arange(ncams)
    cam_combos = []
    for cam in range(2, ncams+1):
        combos = list(combinations(camlist, cam))
        cam_combos.extend(combos)
    ncombos = len(cam_combos)

    # Gather 2D hand locations from all trials
    trialdata_right = sorted(glob.glob(idfolder + '/landmarks/*2Dlandmarks_right.npy'))
    trialdata_left = sorted(glob.glob(idfolder + '/landmarks/*2Dlandmarks_left.npy'))

    # Output directories
    outdir_images = idfolder + '/images/'
    outdir_images_refined = idfolder + '/imagesrefined/'
    outdir_video = idfolder + '/videos_processed/'
    outdir_data3d = idfolder + '/landmarks/'

    # Make output directories if they do not exist (landmarks folder should already exist)
    if not os.path.exists(outdir_images):
        os.mkdir(outdir_images)
    if not os.path.exists(outdir_images_refined):
        os.mkdir(outdir_images_refined)
    if not os.path.exists(outdir_video):
        os.mkdir(outdir_video)
    if not os.path.exists(outdir_data3d + '3d/'):
        os.mkdir(outdir_data3d + '3d/')

    # Low-pass BW filter design
    fc = 5
    fs = 30
    Wn = fc / (fs / 2)
    order = 2
    b, a = signal.butter(order, Wn, btype='low', analog=False, output='ba', fs=None)

    for trialright, trialleft in tqdm(zip(trialdata_right, trialdata_left)):

        # Identify trial name
        filename = os.path.basename(trialright)
        fileparts = filename.split('_2Dlandmarks_right.npy')
        trialname = fileparts[0]
        print(trialname)

        # Load 2D hand location data, fix hand switching (P15 cam orientations slightly odd, so using a different axis)
        data_2d_right = np.load(trialright).astype(float)
        data_2d_left = np.load(trialleft).astype(float)
        if id == 'P05' and visit == 'Fatigue-Flexion':
            switch_hands(data_2d_right, data_2d_left, axis=1)
        else:
            switch_hands(data_2d_right, data_2d_left)

        # Check # of cameras
        if ncams != data_2d_right.shape[0]:
            print('Number of cameras in calibration parameters does not match 2D data.')
            quit()

        # Smooth extraneous / missing 2D data points using a median filter + spline fit
        data_2d_right_spline = smooth2d(data_2d_right, kernel=7, pixeldif=10)
        data_2d_right_unsynced = data_2d_right_spline.copy()

        # Checking synchronization between cameras (ignoring static trials)
        if 'static' not in trialname:
            checksync()

        # Reshape data (only using right hand); treating P17 different here, issue with one trial
        if id == 'P17' and visit == 'Fatigue-Flexion' and trialname == 'trial_post_middle_120':
            data_2d = data_2d_right_unsynced.reshape((ncams, -1, 2))
        else:
            data_2d = data_2d_right_spline.reshape((ncams, -1, 2))

        # Undistort 2D points based on camera intrinsics and distortion coefficients
        # Output is ncams x (nframes x 21 landmarks) x 2-dimension
        data_2d_undistort = np.empty(data_2d.shape)
        for cam in range(ncams):
            data_2d_undistort[cam] = undistort_points(data_2d[cam].astype(float), cam_mats_intrinsic[cam],
                                                      cam_dist_coeffs[cam]).reshape(len(data_2d[cam]), 2)

        # Outputting 3D points
        # Code adapted from aniposelib: https://github.com/lambdaloop/aniposelib/blob/master/aniposelib/cameras.py
        npoints = data_2d_undistort.shape[1]  # nframes x 21
        for combo in cam_combos:
            data3d = np.empty((npoints, 3))
            data3d[:] = np.nan
            cam_mats_extrinsic_sub = [cam_mats_extrinsic[cam] for cam in combo]

            for point in range(npoints):

                # Selecting only from the specific camera combinations for the given frame/landmark point
                subp = data_2d_undistort[combo, point, :]

                # Check how many cameras picked up the landmark for the given frame
                good = ~np.isnan(subp[:, 0])

                # Require at least 2 cameras to have picked up a landmark to triangulate, otherwise keep as nan
                if np.sum(good) >= 2:
                    data3d[point] = triangulate_simple(subp[good], np.array(cam_mats_extrinsic_sub)[good])

            # Reshaping to nframes x 21 landmarks x 3-dimension
            data3d = data3d.reshape((int(len(data3d) / 21), 21, 3))

            # Low-pass BW filter of the 3D data points
            data3d_filt = signal.filtfilt(b, a, data3d, axis=0)

            # Save 3D landmarks as np array
            if len(combo) == ncams:  # Using all cameras
                data3d_use = data3d_filt.copy()

            camcombo_str = ''.join(map(str, combo))
            outdir_data3d_subset = outdir_data3d + '3d/' + camcombo_str + '/'
            if not os.path.exists(outdir_data3d_subset):
                os.mkdir(outdir_data3d_subset)
            np.save(outdir_data3d_subset + trialname + '_3Dlandmarks', data3d_filt)

        # Missing data
        missing = np.count_nonzero(np.isnan(data3d_use))/3
        print('Frames missing: ' + str(missing))

        # Output directories for the specific trial (for visualizations)
        outdir_images_trialfolder = outdir_images_refined + str(trialname) + '/data3d/'
        if not os.path.exists(outdir_images + str(trialname)):
            os.mkdir(outdir_images + str(trialname))
            for cam in range(ncams):
                os.mkdir(outdir_images + trialname + '/cam' + str(cam))
        if not os.path.exists(outdir_images_refined + str(trialname)):
            os.mkdir(outdir_images_refined + str(trialname))
            for cam in range(ncams):
                os.mkdir(outdir_images_refined + trialname + '/cam' + str(cam))
        if not os.path.exists(outdir_images_trialfolder):
            os.mkdir(outdir_images_trialfolder)
        outdir_video_trialfolder = outdir_video + str(trialname)
        if not os.path.exists(outdir_video_trialfolder):
            os.mkdir(outdir_video_trialfolder)

        # Output visualizations
        # Refined 2D labels (note, these are from the unsynced data, so cam frames may be 1-3 frames off)
        vidnames = sorted(glob.glob(idfolder + '/videos/' + trialname + '/*.mp4'))
        visualizelabels(vidnames, data=data_2d_right_unsynced)
        for cam in range(ncams):
            imagefolder = outdir_images_refined + trialname + '/cam' + str(cam)
            createvideo(image_folder=imagefolder, extension='.png', fs=30,
                        output_folder=outdir_video + trialname, video_name='cam' + str(cam) + '_refined.mp4')

        # 3D datapoints
        visualize_3d(data3d_use, save_path=outdir_images_trialfolder + 'frame_{:04d}.png')
        createvideo(image_folder=outdir_images_trialfolder, extension='.png', fs=30,
                    output_folder=outdir_video_trialfolder, video_name='data3d.mp4')

    # Counter
    end = time.time()
    print('Time to run code: ' + str(end - start) + ' seconds')
