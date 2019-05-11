import keras
import numpy as np
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.models import load_model
from keras.applications.mobilenet import preprocess_input

model = load_model('logoclassifier.h5')

def prepare_test_image(file):
    img_path = 'dataset/test/'
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    image.save_img(img_path + file, img_array)
    # print("array shape:", img_array.shape)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

NAMES = ["DyCode", "DycodeX", "Adidas", "Unknown"]

print("\n=========")
preprocessed_image = prepare_test_image('01-dycodex/x0.jpg')
predictions = model.predict(preprocessed_image)
print("Dycode {0:.2f}".format(predictions[0][0]*100))
print("DycodeX {0:.2f}".format(predictions[0][1]*100))
predIdx = np.argmax(predictions, axis=1)[0]
print("Prediction class: {:d} - {}".format(predIdx, NAMES[predIdx]))

print("=========")
preprocessed_image = prepare_test_image('02-adidas/a0.jpg')
predictions = model.predict(preprocessed_image)
print("Adidas {0:.2f}".format(predictions[0][2]*100))
print("DycodeX {0:.2f}".format(predictions[0][1]*100))
predIdx = np.argmax(predictions, axis=1)[0]
print("Prediction class: {:d} - {}".format(predIdx, NAMES[predIdx]))

print("=========")
preprocessed_image = prepare_test_image('00-dycode/d0.jpg')
predictions = model.predict(preprocessed_image)
print("Adidas {0:.2f}".format(predictions[0][2]*100))
print("Dycode {0:.2f}".format(predictions[0][0]*100))
predIdx = np.argmax(predictions, axis=1)[0]
print("Prediction class: {:d} - {}".format(predIdx, NAMES[predIdx]))