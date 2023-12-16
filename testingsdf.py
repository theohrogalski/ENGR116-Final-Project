import keras
import tensorflow as tf
import matplotlib.pyplot
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from numpy import asarray
from PIL import Image
import tensorflow as tf
import keras
from tensorflow import keras
from keras import layers
from keras.models import Sequential
import pathlib
import numpy as np
from tensorflow import expand_dims, nn

class_names=['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

new_model = keras.models.load_model('newtry.h5')
new_model.summary()


#I have to input data as a tensor and then use it as a tensor in the program
img_raw = tf.io.read_file('Images/serve_image.jpeg')

img = tf.image.decode_jpeg(img_raw, channels=3)
img = tf.image.resize(img, [180, 180])
img = img / 255.0
img = tf.expand_dims(img, 0)

new_model.predict(img)

predictions = new_model.predict(img)
score = tf.nn.softmax(predictions[0])

print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], (100 * np.max(score)))
)

print("This image least likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmin(score)], 100-(100 * np.min(score)))
)