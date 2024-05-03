# handmocap

Team: Daanish Mulla, Nigel Majoni, Paul Tilley, Peter Keir

## Background
Multi-camera markerless motion capture of the hand and fingers using MediaPipe (Zhang et al. 2020) and Anipose lib (Karashchuk et al. 2021).

Recorded videos are used to track the hand and predict 3D joint locations of 21 key points of the hand.  The code draws on the work by [Temuge Batpurev](https://temugeb.github.io/python/computer_vision/2021/06/27/handpose3d.html), but extending it to work on 2+ cameras.

https://github.com/DanMulla/handmocap/assets/61323041/90160482-c91f-4d70-b64e-1fd22da0e179

## Steps / Functions

Note: A window will open up asking user to select folder containing data.  This would be the parent folder containing the videos (e.g., /sampledata for example files given).

1. `calibration.py` for intrinsic and extrinsic camera calibration using checkerboard.  With 4 cameras for our resolution and checkerboard, we averaged reprojection errors < 1 pixel.
2. `labels2d.py` for 2D predictions of hand key points from each camera view.  Annotated images will be saved.
3. `triangulation.py` for 3D triangulation of hand key points from calibration and 2D label results.  The triangulation will be done for all camera combinations, but the images/videos saved will be for only the full-camera set.
4. `kinematics.py` to calculate finger joint angles for all camera combinations.
5. `montage.py` for creating a video montage of the raw videos (with or without 2D predictions overlaid) with the 3D kinematics.
6. `trigger.py` for synchronizing the kinematics with other hardware using a custom-designed trigger box that flashes an LED while sending an electrical impulse that can be recorded with other systems (e.g., EMG, Forces).  This function will load the first frame of one of the camera views, ask user prompt to select where in the image the LED will flash, and then automatically detect when the trigger was started by identifying the instance where LED turns on.  Trigger settings will need to be adjusted based on your own custom-designed box (e.g., trigger length).

## Outstanding Issues / Important Considerations

- Most of the code processing time is in saving images (which is turned on by default). The images also take up large disc space. Consider not saving to speed up processing.
- At the moment, both hands (if in view) are tracked, but the triangulation is only done on the right hand as we were recording videos of a single hand. In some images, both hands were in view. We used MediaPipe to identify right vs. left hand, however these predictions are not perfect. We added a temporary solution to fix switching of hands based on where we expected the right hand to be. A general solution needs to be added.
- The triangulation uses a least square method. Anipose has other solutions that are worth considering. On occasion, we noticed that the predictions of uninstructed fingers can be off (e.g., ring finger during little finger flexion). This can likely be alleviated by using Anipose's spatiotemporal regularization or combining a biomechanical modelling approach (OpenSim's IK) to constrain finger lengths.
