import os
from utilities import generateID
from flask import Flask, request
from classify import classify_cpac,classify_ants

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Autism graduation project backend.</p>"

@app.route("/diagnose/", methods=["POST"])
def diagnose():
    dir_path = "../uploads/"
    data = request.json
    file_name = data['file_name']
    model_type = data['model_type']
    
    file_path = dir_path + file_name

    if model_type == 'cpac':
        result = classify_cpac(file_name)
    elif model_type == 'ants':
        result = classify_ants(file_path)
    return result

# @app.route("/cpac/", methods=["POST","GET"])
# def cpac(input_file='test'):
#     result = classify_cpac(input_file)
#     return result

# @app.route("/ants/", methods=["GET"])
# def ants(input_file='test'):
#     # result = classify_ants(input_file)
#     result = '1'
#     return result


app.run()
