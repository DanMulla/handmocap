import glob
from moviepy.editor import VideoFileClip, clips_array
import os
from pathlib import Path
import time
from tkinter import filedialog

# Counter
start = time.time()

# Define working directory
wdir = Path(os.getcwd())

# Open a dialog box to select participant's folder
idfolder = filedialog.askdirectory(initialdir=str(wdir))
id = os.path.basename(os.path.normpath(idfolder))

# Video pathways
rawvideos = idfolder + '/videos/'
processedvideos = idfolder + '/videos_processed/'

# Number of trials
trialfolders = sorted(glob.glob(rawvideos + '*trial*'))
ntrials = len(trialfolders)

for trial in trialfolders:

    # Trial name
    trialname = os.path.basename(trial)

    # Obtain raw videos
    rawvideos = sorted(glob.glob(trial + '/*.mp4'))
    ncams = len(rawvideos)

    # Obtain 3D landmark
    video3d = processedvideos + trialname + '/data3d.mp4'

    # Compile videos (raw + 3D landmarks)
    allvideos = rawvideos.copy()
    allvideos.append(video3d)
    vids = [VideoFileClip(video) for video in allvideos]

    # Check durations consistent across videos and trim to stay at min duration
    durations = [vid.duration for vid in vids]
    vids_trimmed = [vid.subclip(0, min(durations)) for vid in vids]

    # Resize 3D data to appear bigger
    vids_trimmed[-1] = vids_trimmed[-1].resize(1.5)

    # Combine videos together
    top_row = clips_array([vids_trimmed[:-1]])
    bot_row = clips_array([[vids_trimmed[-1]]])
    final_video = clips_array([[top_row], [bot_row]])
    output_path = processedvideos + trialname + '/compilation.mp4'
    final_video.write_videofile(output_path, codec='libx264', fps=30)

# Counter
end = time.time()
print('Time to run code: ' + str(end - start) + ' seconds')
