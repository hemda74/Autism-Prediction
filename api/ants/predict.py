# from main import WIDTH,HEIGHT,FC_LAYERS
import cv2
import numpy as np
from ants.utils import load_class_list,build_finetune_model
from tensorflow.keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input

base_path = "./ants/"
def predict_ants(img_path):

    preprocessing_function = None
    current_model = 'MobileNet'
    dataset = 'ABIDE'
    dropout = 1e-06
    WIDTH = 224
    HEIGHT = 224
    FC_LAYERS = [1024, 1024]
    preprocessing_function = preprocess_input
    if img_path is None:
        ValueError("You must pass an image path when using prediction mode.")

    # Read in your image
    image = cv2.imread(img_path,-1)
    save_image = image
    image = np.float32(cv2.resize(image, (HEIGHT, WIDTH)))
    image = preprocessing_function(image.reshape(1, HEIGHT, WIDTH, 3))

    class_list_file = base_path + "checkpoints/" + current_model + "_" + dataset + "_class_list.txt"

    class_list = load_class_list(class_list_file)
    
    # finetune_model = build_finetune_model(base_model, len(class_list))
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(HEIGHT, WIDTH, 3))
    finetune_model = build_finetune_model(base_model, dropout=dropout, layers=FC_LAYERS, N=len(class_list))
    finetune_model.load_weights(base_path + "checkpoints/" + current_model + "_model_weights.hdf5")

    # Run the classifier and print results
    out = finetune_model.predict(image)

    confidence = out[0]
    class_prediction = list(out[0]).index(max(out[0]))
    class_name = class_list[class_prediction]

    classification = 0
    if class_name[0] == 'ASD':
        classification = 1

    return classification