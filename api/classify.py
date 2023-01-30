import pandas as pd
from ants.predict import predict_ants

predictions_path = "./acerta-abide/predictions.csv"

def classify_cpac(input_file):
    # input_file = "KKI_0050794_rois_cc200.1D"
    f = input_file.split("_")
    if len(f[1]) == 1:
        PID = f[0] + "_" + f[1] + "_" + f[2]
    else:
        PID = f[0] + "_" + f[1]

    patientsDF = pd.read_csv(predictions_path)
    PIDlist = patientsDF["PID"].tolist()
    PYlist = patientsDF["Y"].tolist()

    PIdx = PIDlist.index(PID)
    classification = PYlist[PIdx]
    if classification == 0:
        classification = 1
    else: 
        classification = 0
    return str(classification)

def classify_ants(img_path):
    result = predict_ants(img_path)
    return str(result)