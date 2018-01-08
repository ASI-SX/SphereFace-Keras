import keras
from sphereface_20 import modeldef

img_rows, img_cols = 112, 96 # Resolution of inputs
channel = 3
num_classes = 10572
batch_size = 128
nb_epoch = 1
nb_steps = 28000

model = modeldef(img_rows, img_cols, channel, num_classes)