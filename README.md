# handmocap

## Background
Multi-camera markerless motion capture of the hand and fingers using MediaPipe (Zhang et al. 2020) and Anipose lib (Karashchuk et al. 2021).

Recorded videos are used to track and predict 3D joint locations of 21 key points of the hand.  The code draws on the work by [Temuge Batpurev](https://temugeb.github.io/python/computer_vision/2021/06/27/handpose3d.html), but extending it to work on 2+ cameras.

## Functions

## Outstanding Issues / Important Considerations

- Most of the code processing time is in saving images (which is turned on by default). The images also take up large disc space. Consider not saving to speed up processing.
- At the moment, both hands (if in view) are tracked, but the triangulation is only done on the right hand as we were recording videos of a single hand. In some images, both hands were in view. We used MediaPipe to identify right vs. left hand, however these predictions are not perfect. We added a temporary solution to fix switching of hands based on where we expected the right hand to be.
- The triangulation uses a least square method. Anipose has other solutions that are worth considering. On occasion, we noticed that the predictions of uninstructed fingers can be off (e.g., ring finger during little finger flexion). This can likely be alleviated by using Anipose's spatiotemporal regularization or combining a biomechanical modelling approach (OpenSim's IK) to constrain finger lengths.
