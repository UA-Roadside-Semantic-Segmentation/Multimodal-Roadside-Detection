import dl_lidar.CoralTPU.create_model_tpu as create_model
import tensorflow as tf
import dl_lidar.data_gen as data_gen
import numpy as np
import sys

# if len(sys.argv) != 5:
#     print('Usage: python3 {} weightFile dataset tflite_output scanlines'.format(sys.argv[0]))

# weightFile = sys.argv[1]
# dataset = sys.argv[2]
# output = sys.argv[3]
# scanlines = sys.argv[4]

weightFile = '../reportTraining/weightFiles/allLidarTrainWeights.h5'
dataset = '../reportTraining/H5Files/allLidarTrain.hdf5'
output = 'reportModels/allLidarTrain'
scanlines = 64

def representative_dataset_gen():
    for _ in range(1000):
        data = next(val_gen)
        print("{} - {}".format(_, data[0].shape))
        yield [data[0].astype(np.float32)]

model = create_model.create_model(scanlines)
model.load_weights(weightFile)

val_gen = data_gen.data_gen(dataset, 'training', 2, 1, augment=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8
converter.inference_output_type = tf.int8  # or tf.uint8
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()

with open(output + '.tflite', 'wb') as f:
  f.write(tflite_quant_model)