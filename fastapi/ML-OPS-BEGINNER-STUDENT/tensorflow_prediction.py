from numpy.lib.npyio import load
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
import os

def load_image(filename):
    # load the image
    img = load_img(filename, target_size=(224, 224))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 3 channels
    img = img.reshape(1, 224, 224, 3)
    # center pixel data
    img = img.astype('float32')
    img = img - [123.68, 116.779, 103.939]
    return img

# load the model
model = tf.keras.models.load_model("C:/Users/rdescobedo/Desktop/Carpetas/MLOps/fastapi/ML-OPS-BEGINNER-STUDENT/final_model")

img = load_image("C:/Users/rdescobedo/Desktop/Carpetas/MLOps/fastapi/ML-OPS-BEGINNER-STUDENT/graham2.jpg")
prediction = model.predict(img)

if prediction [0][0] > 0.5: #0-0.5 = cat & 0.5-1 = dog
    print("Pic of dog")
else: 
    print("Pic of cat")
