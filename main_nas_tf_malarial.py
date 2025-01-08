
# # Malarial cell classification using CNN and data augmentation
# It's a binary classification problem where we have two sets of images : parasitized and uninfected.
# have some images for training but not enough to have robust segmentation.
# We'll augment this data and use in the model training.
import tensorflow as tf
import keras
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
print("Num CPUs Available: ", len(tf.config.list_physical_devices('CPU')))

from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image
import scipy
import nni
import nni 
from keras.callbacks import Callback


def get_data():
    batch_size = 32
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
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
    return train_generator, validation_generator, batch_size



 # Defining the CNN model

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
#from keras import backend as K

num_classes = 2

def build_model(params):
  model = Sequential([
      Conv2D(params['filter_size_c1'],params['kernel_size_c1'],activation = 'relu'),
      Conv2D(params['filter_size_c2'],params['kernel_size_c2'], activation = 'relu'),
      MaxPooling2D(),
      Flatten(),
      Dense(params['nb_units'], activation = 'relu'),
      Dense(1, activation = 'sigmoid')])
  
  optimizer = keras.optimizers.RMSprop(learning_rate= params['learning_rate'])
  model.compile(optimizer= optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
  return model

# report intermediate results 
class SendMetrics(Callback): 
    def on_epoch_end(self, epoch, logs= {}):
        nni.report_intermediate_result(logs['val_accuracy'])


# Notice that the input is an image and the output is a binary number. 

# ## Fitting the model

#We can now use these generators to train our model. 

def run(params, train_generator, validation_generator,batch_size):
    model = build_model(params)
    model.fit(
            train_generator,
            epochs=5,
            verbose=1,
            validation_data=validation_generator,
            validation_steps=800 // batch_size)
    train_loss, train_acc, validation_loss, test_acc = eval(model, train_generator,validation_generator)
    nni.report_final_result(test_acc)
    print(test_acc)
    

# print(len(X_train))
#model.save('malaria_augmented_model.h5')  # always save your weights after training or during training

# evaluating the model

def eval(model, train_generator,validation_generator):

    validation_loss, test_acc = model.evaluate(validation_generator, steps=16)
    print('Test: %.3f' % ( test_acc))
    return validation_loss, test_acc



if __name__ == '__main__':
   try:
# get parameters from nni
    params = nni.get_next_parameters() # will choose parameters from search space 
    train_generator, validation_generator, batch_size = get_data()
    run(params,train_generator, validation_generator,batch_size)
   except Exception:
      raise


