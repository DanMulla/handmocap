import cv2 as cv
import glob
import numpy as np
import os
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog
import toml


def groups(arrayinput, *, threshold, win_size):
    """
    Identifying all indices of an input array where the values exceed a given
    threshold consecutively for at least a given window size.
    Code from here: https://stackoverflow.com/questions/74301529/how-to-get-the-indices-of-at-least-two-consecutive-values-that-are-all-greater-t

    :param arrayinput: Input array.
    :param threshold: Threshold value exceeding which is flagged.
    :param win_size: Number of consecutive indices where threshold met.
    :return: Indices meeting threshold criteria for given window size.
    """

    conv = np.convolve((arrayinput >= threshold).astype(int), [1] * win_size, mode="valid")
    indices_start = np.where(conv == win_size)[0]
    indices = [np.arange(index, index + win_size) for index in indices_start]
    indices = np.unique(indices)

    return indices


def click_event(event, x, y, flags, params):
    """
    Identifying left mouse button clicks and returning position of click.
    Code adapted from here: https://stackoverflow.com/questions/28327020/opencv-detect-mouse-position-clicking-over-a-picture
    """

    global led

    # Check for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:

        # Add mouse click coordinates to led position
        led.append((x, y))

        # Display coordinates
        cv.putText(frame, str(x) + ',' + str(y),
                   (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv.circle(img=frame, center=(x, y), radius=3,
                  color=(0, 0, 255), thickness=-1)

        # Display image
        cv.imshow('image', frame)


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

    # Gather trial folders
    trialfolders = sorted(glob.glob(idfolder + '/videos/*trial*'))

    # Trigger settings
    # threshold set at 10% as LED light was quite small and want to pick up as soon as it lit up
    # win_size set at 8 frames based on 30 fps and our trigger set to turn on for 500 ms
    # note: anyone in lab using the custom-designed trigger box, led light fades as battery gets old (be cautious and replace!)
    box_area = 100   # Square area (in pixels) centered on trigger led
    box_size = int(np.sqrt(box_area) / 2)
    threshold = 0.10  # Fraction of pixels lit within box to denote if LED is turned on
    win_size = 8    # Number of consecutive frames LED turned on

    # Storage for trigger results
    trigger_output = {}

    for trial in trialfolders:

        # Obtain video (only using one angle/camera for each trial)
        trialname = os.path.basename(trial)
        trialvids = sorted(glob.glob(trial + '/*.mp4'))
        video = trialvids[-1]

        # Load capture and get video settings
        cap = cv.VideoCapture(video)
        length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(3))
        height = int(cap.get(4))

        # Create background subtractor object
        bs = cv.createBackgroundSubtractorMOG2()

        # Pre-allocate storage for all frames and trigger led position
        frames_all = np.zeros([height, width, length])
        led = []

        # Frame iterator
        count = 0

        while True:

            # Read the next frame
            ret, frame = cap.read()

            # Break if no frames found
            if not ret:
                break

            # Show first frame and prompt user to click on trigger led
            if count == 0:
                cv.putText(frame, trialname + ': Click on trigger light and press enter.',
                           (50, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                cv.imshow('image', frame)
                cv.setMouseCallback('image', click_event)
                cv.waitKey(0)
                led = np.array(led)
                cv.destroyAllWindows()

            # Background subtraction based on mixture of gaussians (Subtractor MOG2)
            # Code from here: https://hackmd.io/@lKuOpplzSUWLhLim2Z7ZJw/SkL-qU2Wh#Subtraction-using-Subtractor-MOG2
            # note: there is some initialization here, so trigger didn't get detected if it was hit within first 1 second of video
            fgmask = bs.apply(frame)
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
            fgmask = cv.morphologyEx(fgmask, cv.MORPH_OPEN, kernel)

            # Display image
            # cv.imshow('frame', fgmask)
            # if cv.waitKey(1) & 0xFF == ord('q'):
            #    break

            # Store frames
            frames_all[:, :, count] = fgmask

            # Update iterator
            count += 1

        # Clear windows
        cap.release()
        cv.destroyAllWindows()

        # Identify frames where LED light was turned on (based on trigger settings)
        pixels_on = np.mean(frames_all[led[0][1] - box_size:led[0][1] + box_size,
                            led[0][0] - box_size:led[0][0] + box_size], axis=(0, 1)) / 255
        led_on = groups(pixels_on, threshold=threshold, win_size=win_size)
        nframes_on = led_on.size

        # Check if LED light was detected at any point
        if nframes_on == 0:
            startframe = -1
            print(trialname + ': Trigger light not detected.')
        else:
            startframe = int(led_on[0])  # Need int here otherwise saved as string in toml file
            print(trialname + ': Trigger turns on at frame # ' + str(startframe) + ' and was on for ' + str(nframes_on) + ' frames.')

        # Storing trigger output
        trigger_results = {trialname: {'framestart': startframe, 'nframes_on': nframes_on}}
        trigger_output.update(trigger_results)

    # Save trigger results to toml file
    with open(idfolder + '/trigger.toml', 'w') as file:
        toml.dump(trigger_output, file)

    # Counter
    end = time.time()
    print('Time to run code: ' + str(end - start) + ' seconds')
