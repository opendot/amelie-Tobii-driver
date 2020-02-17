# Amelie
 Amelie Suite is a set of software co-designed for people suffering from Rett's syndrome characterized by an innovative way of interaction between care-giver and care-receiver, both equipped with an instrument on their own device, and through the use of an eyetracker (which allows you to track the look of the subject and to determine which point on the screen is watching).


Amelie is an open source software and accessible to everyone, co-designed by designers, developers, university researchers, families and therapists. Amelie promotes communication and cognitive enhancement for learning and improving interaction and understanding skills.
The system integrates different technologies (mobile applications, cloud services and adaptive algorithms) to provide an innovative, comprehensive and easy-to-use service. 


The software was born from an idea of Associazione Italiana Rett - AIRETT Onlus and Opendot S.r.l., and was designed and developed by Opendot S.r.l., with the essential contribution of Associazione Italiana Rett - AIRETT Onlus.

This repository hosts the Tobii eyetracker driver to be used on the local PCs.


# Amelie-driver
driver for the Tobii eye tracker

- reads the eyetracker incoming data
- sends the gaze position at a specified time interval for real-time visualization purposes
- detects fixations using two distinct methods (trained and untrained)
- saves its state on every relevant change on a json file, in order to be restored in case of crashes

## dependencies

The driver needs the following dependencies:
- boost liraries: https://www.boost.org/
- openBLAS: https://www.openblas.net/
- TOBII eyetracker stream engine c++ (nuGet package): https://github.com/Tobii/stream_engine
- Armadillo: http://arma.sourceforge.net/ 
- Eigen: http://eigen.tuxfamily.org/index.php?title=Main_Page
- ML pack: https://www.mlpack.org/
- websocketpp: https://github.com/zaphoyd/websocketpp
- json: https://github.com/nlohmann/json

most of them can be installed through [vcpkg](https://github.com/Microsoft/vcpkg). You can find a walkthrough about installing MLpack om Windows [here](https://keon.io/mlpack-on-windows/)

## Detecting fixations

Two distinct methods are used, according to the user's preferences:

**Untrained**

takes two parameters from the mobile application, fixation time (default 600ms) and fixation radius (default 0.05, 5% of the screen width). 

pseudocode:
```
Fixation radius Fr
Fixation time Ft

For each gaze data point Pi:
  if reference point R is not set:
    set R = Pi
    set Now = current time
  else:
    if distance(R,Pi) < Fr:
      if current time - Now >= Ft:
        fixation detected at point R
        reset
      else:
        continue
    else:
      reset
```

**Trained**


The algorithm is described in [this paper](https://www.nature.com/articles/s41598-017-17983-x). A ~1 minute long video must be provided to train the Hidden Markov Model that predicts the gaze pattern.


## Websocket Ports and messaging

The websocket server uses port 4000. A list of the accepted messages can be found in the [webspcket.hpp file](https://github.com/dot-dot-dot/airett-driver/blob/master/websocket.hpp)



