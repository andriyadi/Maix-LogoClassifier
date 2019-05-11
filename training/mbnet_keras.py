import keras
import numpy as np
from keras import backend as K
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.applications import imagenet_utils
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input

# Adjust these
NUM_CLASSES = 4
NAMES = ["DyCode", "DycodeX", "Adidas", "Unknown"]
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
TRAINING_DIR = 'dataset/train'
VALIDATION_DIR = 'dataset/test'

base_model=keras.applications.mobilenet.MobileNet(input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, 3), alpha = 0.75,depth_multiplier = 1, dropout = 0.001,include_top = False, weights = "imagenet", classes = 1000)

# Additional Layers
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(100,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dropout(0.5)(x)
x=Dense(50, activation='relu')(x) #dense layer 3
preds=Dense(NUM_CLASSES, activation='softmax')(x) #final layer with softmax activation


model=Model(inputs=base_model.input,outputs=preds)

for i,layer in enumerate(model.layers):
    print(i,layer.name)

# or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:86]:
    layer.trainable=False
for layer in model.layers[86:]:
    layer.trainable=True

train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
# train_datagen = ImageDataGenerator( rescale = 1./255,
#                                     rotation_range=45,
#                                     width_shift_range=0.1,
#                                     height_shift_range=0.1,
#                                     shear_range=0.1,
#                                     zoom_range=[0.9, 1.2],
#                                     horizontal_flip=True,
#                                     vertical_flip=False,
#                                     fill_mode='constant',
#                                     brightness_range=[0.7, 1.3])

train_generator=train_datagen.flow_from_directory(TRAINING_DIR,
                                                 target_size=(IMAGE_WIDTH,IMAGE_HEIGHT),
                                                 color_mode='rgb',
                                                 batch_size=5,
                                                 class_mode='categorical', shuffle=True, 
                                                #  save_to_dir='dataset/gen', 
                                                #  save_prefix='gen-', 
                                                #  save_format='jpeg'
                                                 )

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
# validation_datagen = ImageDataGenerator(rescale = 1./255,
#                                         rotation_range=45,
#                                         zoom_range=[0.9, 1.2],
#                                         shear_range=0.1,)
validation_generator = validation_datagen.flow_from_directory( 	VALIDATION_DIR,
								target_size=(IMAGE_WIDTH,IMAGE_HEIGHT),
    								color_mode='rgb',
    								batch_size=5,
								class_mode='categorical'
)

model.summary()
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer, loss function will be categorical cross entropy, evaluation metric will be accuracy

step_size_train = (train_generator.n//train_generator.batch_size)
validation_steps = (train_generator.n//train_generator.batch_size)
model.fit_generator(generator=train_generator, 
                    steps_per_epoch=step_size_train, 
                    epochs=50, 
                    validation_data = validation_generator, 
                    validation_steps = validation_steps,
                    verbose = 1)

model.save('logoclassifier.h5')

#model.load_weights('logoclassifier.h5')

def prepare_test_image(file):
    img_path = 'dataset/test/'
    img = image.load_img(img_path + file, target_size=(IMAGE_WIDTH, IMAGE_HEIGHT))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

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
