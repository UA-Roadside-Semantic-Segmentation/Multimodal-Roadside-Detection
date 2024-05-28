## Dataset Utilities
| File | Description |
|------|-------------|
| `h5Creator.py` | Takes list of files and splits it into the TTV sets, saving them to a single h5 files. Function for images and PCD's |
| `createPCDDataset.py` | Given a directory of raw and labeled PCD files, will create the h5 file for it, using H5Creator |
| `createImgDataset.py` | Given a directory of raw and labeled image files, will create the h5 file for it, using H5Creator |
| `dataset.py` | Helper functions for creating datasets |
| `find_matching_files.py` | Correlates .jpg and .pcd files by their timestamps given in `camera_timestamps.txt` and `lidar_timestamps.txt`

## Data Ingest Utilities
| File | Description |
|------|-------------|
| `dual_return_fix.cpp` | C++ code to pull the single-return strongest point cloud from a dual-return point cloud |
| `fix_timestamps.py` | Used for calculating corrected file timestamps and reading/writing mtimes |
| `rename_train_images.py` | Used for initial rename of train data collection files |
| `rename_truck_images.py` | Used for initial rename of truck data collection files. Can also be used for train data with a one-line change |
| `pcdConverter.cpp`<br>in the [lidar-drivers](https://gitlab.com/camgian-research/lidar-drivers) repo | Used to convert collected .pcd files to the ASCII format compatible with the rest of the toolchain |
