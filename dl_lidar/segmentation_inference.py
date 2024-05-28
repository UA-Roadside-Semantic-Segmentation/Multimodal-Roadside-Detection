import tensorflow as tf
from dl_lidar import create_model
import os
from dl_lidar import data_gen
import gzip
import numpy as np
import matplotlib.pyplot as plt

m = tf.keras.metrics.MeanIoU(num_classes=3)

def MeanIoU(labels, prediction):
    global m
    #print("Labels",labels.shape)
    #print("Predictions",prediction.shape)
    # voidPredBool = prediction[:,:,:,0] > 0.5
    rockPredBool = prediction[:,:,:,1] > 0.5
    # craterPredBool = prediction[:,:,:,2] > 0.5
    # voidLabelBool = labels[:, :, :, 0] > 0.5
    rockLabelBool = labels[:, :, :, 1] > 0.5
    # craterLabelBool = labels[:, :, :, 2] > 0.5

    rockIntersection = tf.math.reduce_sum(tf.cast(tf.math.logical_and(rockLabelBool,rockPredBool),tf.float32))
    rockUnion = tf.math.reduce_sum(tf.cast(tf.math.logical_or(rockLabelBool,rockPredBool),tf.float32))
    #
    # craterIntersection = tf.math.reduce_sum(tf.cast(tf.math.logical_and(craterLabelBool, craterPredBool), tf.float32))
    # craterUnion = tf.math.reduce_sum(tf.cast(tf.math.logical_or(craterLabelBool, craterPredBool), tf.float32))
    #
    # voidIntersection = tf.math.reduce_sum(tf.cast(tf.math.logical_and(voidLabelBool, voidPredBool), tf.float32))
    # voidUnion = tf.math.reduce_sum(tf.cast(tf.math.logical_or(voidLabelBool, voidPredBool), tf.float32))
    #
    tf.print(tf.math.divide(rockIntersection,rockUnion))
    # tf.print(tf.math.divide(craterIntersection, craterUnion))
    # tf.print(tf.math.divide(voidIntersection, voidUnion))
    #print(rockLabels.shape)

    y = tf.math.argmax(labels, axis=3)
    y = tf.cast(y, dtype=tf.float32)

    x = tf.math.argmax(prediction, axis=3)
    x = tf.cast(x, dtype=tf.float32)

    m.reset_states()
    m.update_state(x, y)

    return m.result()


f_data = gzip.GzipFile('test_data.npy.gz', 'r')
f_labels = gzip.GzipFile('test_labels.npy.gz', 'r')

data = np.load(f_data)
labels = np.load(f_labels)


tf.keras.backend.set_learning_phase(0)
model = create_model.create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0001), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy', MeanIoU])

model_path = 'weightsNew.h5'
model.load_weights(model_path)

print(data[0].shape)
# input_tensor = np.empty((0, 16, 2048, 4))
# input_tensor = data[0]
# input_tensor = input_tensor[np.newaxis, ...]
# print(input_tensor.shape)
#
# truth_tensor = tf.one_hot(labels[0].astype(np.int32), 3)
# truth_tensor = truth_tensor[np.newaxis, ...]

# output_tensor = model.predict(input_tensor)
gen = data_gen.data_gen('validation_data.npy.gz', 'validation_labels.npy.gz', 3, 1, mirror_horizontal=True, add_noise=True)
input_tensor, labels_gen = next(gen)
output_tensor = model.predict(x=input_tensor)

# print('evaluating')
# model.evaluate(x=data_gen.data_gen('validation_data.npy.gz', 'validation_labels.npy.gz', 3, 4))
# print('done evaluating')

output_tensor = tf.math.argmax(output_tensor, axis=3).numpy()
print(output_tensor.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_facecolor((0,0,0))
x = np.reshape(input_tensor[0,:,:,0], 32768)
y = np.reshape(input_tensor[0,:,:,1], 32768)
z = np.reshape(input_tensor[0,:,:,2], 32768)
c = np.reshape(output_tensor[0,:,:], 32768)

ax.scatter3D(x, y, z, c=c, cmap='Greens', s=1)
ax.autoscale(enable=False,axis='both')
ax.set_xbound(-12.5,12.5)
ax.set_ybound(-6,6)
ax.set_zbound(-3,3)
plt.show()