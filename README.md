# UA Roadside Semantic Segmentation

## Overview
This work proposes a portable, multimodal semantic segmentation system for roadside vehical detection. An RGB monocular camera (FLIR Blackfly S GigE BFS-PGE-31S4C-C) and multiple LiDAR (Velodyne Puck, Velodyne Puck Hi-Res, Velodyne 32MR, Ouster OS-1, and Ouster OS-2) models were developed to detect semi-trucks and train cars while ignoring other motor vehicles as part of a targeted vehicle tracking system. A depiction of the roadside system is shown below. 

![Overview](https://github.com/UA-Roadside-Semantic-Segmentation/Multimodal-Roadside-Detection/blob/main/Figures/overview.PNG)


This repository contains all of the packages required for the camera and LiDAR semantic segmentation models.

## Installation
To install the packages so that you can use them together (ex. 
call a dl_utils function from a dl_lidar module), simply run 
`./setup_repository.sh` from the root directory.

If you are using PyCharm to run prorams, you do not need to install 
anything, as Pycharm will automatically add the packages to the PYTHONPATH.

## Packages
### dl_camera
This package contains all files that are specific to deep learning for
cameras (ex. image training scripts)

### dl_lidar
This package contains all files that are specific to deep learning for
lidars for the project (ex. pcd training scripts)

### dl_utils
This package contains anything that may be useful in other deep learning 
projects (ex. visualizers, dataset management, etc.)
