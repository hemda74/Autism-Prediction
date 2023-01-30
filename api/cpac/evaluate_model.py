"""
  nn_evaluate.py --whole cc200
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from docopt import docopt
from train_model import nn
from utils import (load_phenotypes, format_config, hdf5_handler,
                   reset, to_softmax, load_ae_encoder, load_fold)
from sklearn.metrics import confusion_matrix

def nn_results(hdf5, exp, size1, size2):

    exps = hdf5["experiments"][exp]
    N = 2
    results = []
    patient_ids = []
    predictions = []
    true_vals = []

    for fold in exps:

        experiment_cv = format_config("{experiment}_{fold}", {
            "experiment": exp,
            "fold": fold,
        })

        XTrain, YTrain, XValid, YValid, XTest, YTest, currPID = load_fold(hdf5["patients"], exps, fold)

        YTest = np.array([to_softmax(N, y) for y in YTest])

        ae1_model_path = format_config("./data/models/{experiment}_autoencoder-1.ckpt", {
            "experiment": experiment_cv,
        })
        ae2_model_path = format_config("./data/models/{experiment}_autoencoder-2.ckpt", {
            "experiment": experiment_cv,
        })
        nn_model_path = format_config("./data/models/{experiment}_mlp.ckpt", {
            "experiment": experiment_cv,
        })

        try:

            model = nn(XTest.shape[1], N, [
                {"size": 1000, "actv": tf.nn.tanh},
                {"size": 600, "actv": tf.nn.tanh},
            ])

            init = tf.global_variables_initializer()
            with tf.Session() as session:

                session.run(init)
                saver = tf.train.Saver(model["params"])
                saver.restore(session, nn_model_path)

                output = session.run(
                    model["output"],
                    feed_dict={
                        model["input"]: XTest,
                        model["dropouts"][0]: 1.0,
                        model["dropouts"][1]: 1.0,
                    }
                )

                YPred = np.argmax(output, axis=1)
                YTrue = np.argmax(YTest, axis=1)
                
                patient_ids.extend(currPID)
                predictions.extend(YPred.tolist())
                true_vals.extend(YTrue.tolist())

                [[TN, FP], [FN, TP]] = confusion_matrix(YTrue, YPred, labels=[0, 1]).astype(float)
                accuracy = (TP+TN)/(TP+TN+FP+FN)
                specificity = TN/(FP+TN)
                precision = TP/(TP+FP)
                sensivity = recall = TP/(TP+FN)
                fscore = 2*TP/(2*TP+FP+FN)

                results.append([accuracy, precision, recall, fscore, sensivity, specificity])
        finally:
            reset()

    dictionary = {'PID': patient_ids, 'Y': predictions,'Y_true':true_vals}  
    dataframe = pd.DataFrame(dictionary) 
    dataframe.to_csv('predictions2.csv',index=False)
    return [exp] + np.mean(results, axis=0).tolist()

if __name__ == "__main__":

    reset()

    arguments = docopt(__doc__)

    pd.set_option("display.expand_frame_repr", False)

    pheno_path = "./data/phenotypes/Phenotypic_V1_0b_preprocessed1.csv"
    pheno = load_phenotypes(pheno_path)

    hdf5 = hdf5_handler(bytes("./data/abide.hdf5",encoding="utf-8"), "a")

    cc200der = ["cc200"]
    derivatives = [derivative for derivative
                   in arguments["<derivative>"]
                   if derivative in cc200der]

    exps = []

    for d in derivatives:

        config = {"derivative": d}

        if arguments["--whole"]:
            exps += [format_config("{derivative}_whole", config)]


    # First autoencoder 
    code_size_1 = 1000

    # Second autoencoder 
    code_size_2 = 600

    results = []

    exps = sorted(exps)

    for experiment in exps:
        results.append(nn_results(hdf5, experiment, code_size_1, code_size_2))

    cols = ["Exp", "Accuracy", "Precision", "Recall", "F-score",
            "Sensivity", "Specificity"]
    dataFrame = pd.DataFrame(results, columns=cols)

    dataFrame[cols].sort_values(["Exp"])
    dataFrame[cols].reset_index()
    print(dataFrame[cols])
    
        
