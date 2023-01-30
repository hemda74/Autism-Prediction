from __future__ import print_function
from keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
import os
import utils

BatchSize = 32
width = 224
height = 224
layers = [1024, 1024]
trainDir = "ABIDE/train/"
validDir = "ABIDE/val/"
preprocFun = preprocess_input
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(height, width, 3))

if not os.path.isdir("checkpoints"):
    os.makedirs("checkpoints")

train_datagen =  ImageDataGenerator(
    preprocessing_function= preprocFun,
    rotation_range= 0.0,
    shear_range= 0.0,
    zoom_range= 0.0,
    horizontal_flip= True,
    vertical_flip= False
)
val_datagen = ImageDataGenerator(preprocessing_function=preprocFun)

train_generator = train_datagen.flow_from_directory(trainDir, target_size=(height, width), batch_size=BatchSize)

validation_generator = val_datagen.flow_from_directory(validDir, target_size=(height, width), batch_size=BatchSize)

class_list = utils.get_subfolders(trainDir)

utils.save_class_list(class_list, model_name="MobileNet", dataset_name="ABIDE")
finetune_model = utils.build_finetune_model(base_model, dropout=1e-6, layers=layers, N=len(class_list))

adam = Adam(lr=0.00001)

finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])

num_train_images = utils.get_num_files(trainDir)
num_val_images = utils.get_num_files(validDir)

def lr_decay(epoch):
    if epoch%20 == 0 and epoch!=0:
        lr = K.get_value(base_model.optimizer.lr)
        K.set_value(base_model.optimizer.lr, lr/2)
        print("LR changed to {}".format(lr/2))
    return K.get_value(base_model.optimizer.lr)


learning_rate_schedule = LearningRateScheduler(lr_decay)

filepath="./checkpoints/MobileNet_model_weights.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor=["acc"], verbose=1, mode='max')
callbacks_list = [checkpoint]
history = finetune_model.fit(train_generator, epochs= 15, workers=8, steps_per_epoch=num_train_images // BatchSize,
    validation_data=validation_generator, validation_steps=num_val_images // BatchSize, shuffle=True, callbacks=callbacks_list)
