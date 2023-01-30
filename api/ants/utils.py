from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.models import  Model
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import glob
import os, csv

def save_class_list(class_list, model_name, dataset_name):
    class_list.sort()
    target=open("./checkpoints/" + model_name + "_" + dataset_name + "_class_list.txt",'w')
    for c in class_list:
        target.write(c)
        target.write("\n")

def load_class_list(class_list_file):
    class_list = []
    with open(class_list_file, 'r') as csvfile:
        file_reader = csv.reader(csvfile)
        for row in file_reader:
            class_list.append(row)
    class_list.sort()
    return class_list

def get_subfolders(dir):
    subfolders = os.listdir(dir)
    subfolders.sort()
    return subfolders

def get_num_files(dir):
    if not os.path.exists(dir):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(dir):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt

def build_finetune_model(base_model, dropout, layers, N):
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)
    for fc in layers:
        x = Dense(fc, activation='relu')(x) # New FC layer, random init
        x = Dropout(dropout)(x)

    predictions = Dense(N, activation='softmax')(x) # New softmax layer
    
    finetune_model = Model(inputs=base_model.input, outputs=predictions)

    return finetune_model

def plot_training(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.show()