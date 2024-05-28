import tflite_runtime.interpreter as tflite
from time import time
import numpy as np
import h5py
import sys

if len(sys.argv) != 4:
  print('Usage: python3 {} dataset model inferenceOutput'.format(sys.argv[0]))
  sys.exit()

dataset = sys.argv[1]
hf = h5py.File(dataset, 'r')
dataset_size = hf['testing_data'].shape

iter = dataset_size[0]

interpreter = tflite.Interpreter(model_path=sys.argv[2],
                                 experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
input_data = np.empty((1,16,2048,4))
input_data[0] = hf['testing_data'][0]

interpreter.set_tensor(input_details[0]['index'], np.float32(input_data))
print('0 of {}'.format(iter))
start_time = time()
interpreter.invoke()
end_time = time()
output_data = interpreter.get_tensor(output_details[0]['index'])

initTime = end_time - start_time
totalTime = initTime

inferred_hf = h5py.File(sys.argv[3], 'a')

inferred_dset = inferred_hf.create_dataset('inferred', maxshape=((None,) + output_data[0].shape), shape=((1,) + output_data[0].shape), compression='gzip')
inferred_dset[:] = output_data
print(inferred_dset.shape)

for i in range(1, iter):
  print('{} of {}'.format(i, iter))
  input_data[0] = hf['testing_data'][i]
  interpreter.set_tensor(input_details[0]['index'], np.float32(input_data))
  start_time = time()
  interpreter.invoke()
  end_time = time()
  totalTime += end_time - start_time
  output_data = interpreter.get_tensor(output_details[0]['index'])

  inferred_dset.resize(inferred_dset.shape[0] + 1, axis=0)
  inferred_dset[-1:] = output_data
  print(inferred_dset.shape)

print('Total time to run inference on {} scans: {} s '.format(iter, totalTime))
print('Average time per scan: {} s'.format(totalTime/iter))
print('Time for first inference: {}'.format(initTime))
print('Average time excluding first inference: {}'.format((totalTime - initTime)/(iter-1)))
hf.close()
inferred_hf.close()

