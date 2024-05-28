from utils import dataset
import dataset_viewer
import numpy as np
import cv2

# filepaths = ['../blackfly_000_00000.jpg', '../blackfly_000_00001.jpg', '../blackfly_000_00002.jpg']
# img_array = dataset.img_list_to_ndarray(filepaths)
# dataset.save_ndarray_as_npy_gz(img_array, './test.npy.gz')
#
# new_array = dataset.load_npy_gz('test.npy.gz')
# cv2.imshow('t', new_array[2])
# cv2.waitKey(0)
# cv2.destroyAllWindows()

filepaths_pcd = ['hi-res_000_00000.pcd', 'hi-res_000_00001.pcd']
data_pcd = dataset.pcd_list_to_ndarray(filepaths_pcd)
dv = dataset_viewer.DatasetViewer(data_pcd, np.zeros((2, 16, 2048, 1)), 't')
dv.show()