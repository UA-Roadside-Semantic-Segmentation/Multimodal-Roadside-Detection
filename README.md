# Ricks Camera/Lidar Deep Learning

## Overview
This repository contains all of the packages for deep learning.

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
lidars for the project(ex. pcd training scripts)

### dl_utils
This package contains anything that may be useful in other deep learning 
projects (ex. visualizers, dataset management, etc.)