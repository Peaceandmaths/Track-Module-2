
# # Malarial cell classification using CNN and data augmentation
# It's a binary classification problem where we have two sets of images : parasitized and uninfected.
# have some images for training but not enough to have robust segmentation.
# We'll augment this data and use in the model training.
import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image
import scipy


datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


# ## Recall : Single image augmentation

# path = r"C:\Users\golub\PycharmProjects\pythonProject\Track module 2\cell_images\Parasitized\C37BP2_thinF_IMG_20150620_133205a_cell_88.png"
# img = load_img(path)  
# # uses Pillow in the backend, so need to convert to array

# x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
# x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# # the .flow() command below generates batches of randomly transformed images
# # and saves the results to the `preview/` directory
# i = 0
# for batch in datagen.flow(x, batch_size=1,  
#                           save_to_dir=r"C:\Users\golub\PycharmProjects\pythonProject\Track module 2\augmented", save_prefix='aug', save_format='png'):
#     i += 1
#     if i > 20:
#         break  # otherwise the generator would loop indefinitely


# # Defining the CNN model

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras import backend as K


SIZE = 150
batch_size = 1
###2 conv and pool layers. with some normalization and drops in between.

INPUT_SHAPE = (SIZE, SIZE, 3)   #change to (SIZE, SIZE, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=INPUT_SHAPE))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

print(model.summary())    


# Notice that the input is an image and the output is a binary number. 

# ## Fitting the model


#Let's prepare our data. We will use .flow_from_directory() 
#to generate batches of image data (and their labels) 
#directly from our png in their respective folders.

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255, 
        rotation_range=45,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# this is the augmentation configuration we will use for validation:
# only rescaling. But we can try other operations

validation_datagen = ImageDataGenerator(rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data

cell_images_path =  '/home/golubeka/Trackmodule2/Malarial_cells/train'  # this is the input directory
train_generator = train_datagen.flow_from_directory(cell_images_path,
        target_size=(150, 150),  # all images will be resized to 64x64
        batch_size=batch_size,
        class_mode='binary')  
# since we use binary_crossentropy loss function, we need binary labels

# this is a similar generator, for validation data
validation_generator = validation_datagen.flow_from_directory(
       '/home/golubeka/Trackmodule2/Malarial_cells/val' ,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')

# #Add checkpoints 
# #Add Callbacks, e.g. ModelCheckpoints, earlystopping, csvlogger.
# from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger

# #ModelCheckpoint callback saves a model at some interval.
# filepath="saved_models/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5" #File name includes epoch and validation accuracy.
# #Use Mode = max for accuracy and min for loss.
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

# #https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
# early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
# #This callback will stop the training when there is no improvement in
# # the validation loss for three consecutive epochs.

# #CSVLogger logs epoch, acc, loss, val_acc, val_loss
# log_csv = CSVLogger('my_logs.csv', separator=',', append=False)

# callbacks_list = [checkpoint, early_stop, log_csv]

#We can now use these generators to train our model. 
history = model.fit_generator(
        train_generator,
#The 2 slashes division return rounded integer
        epochs=5,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=800 // batch_size)

# print(len(X_train))
model.save('malaria_augmented_model.h5')  # always save your weights after training or during training

# Because we use augmentation, in stead of mofel.fit we apply model.fit_generator.
# Input data is coming from the train_generator, that's why when we're using the Data Augmentation,
# we don't save it to our local drive and upload  it back into the program, it takes extra memory.
# Data can be generated on the fly. If we use Data Augmentation, the input file that is used for augmentation
# will never be passed into model.fit, it's the generated data.
# The original file will not be part of our model.fit. Only the generated augmented images are part of the fit.

# evaluating the model

train_loss, train_acc = model.evaluate_generator(train_generator, steps=16)
validation_loss, test_acc = model.evaluate_generator(validation_generator, steps=16)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))

from matplotlib import pyplot as plt
# plot training history
print("Values stored in history are ... \n", history.history)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()

"""
#To continue training, by modifying weights to existing model.
#The saved model can be reinstated.

from keras.models import load_model
new_model = load_model('malaria_augmented_model.h5')
results = new_model.evaluate_generator(validation_generator)
print(" validation loss and accuracy are", results)

new_model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,    #The 2 slashes division return rounded integer
        epochs=5,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        callbacks=callbacks_list)

model.save('malaria_augmented_model_updated.h5') 

"""



