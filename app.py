from flask import Flask, request, render_template
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
from flask_cors import CORS, cross_origin

app = Flask(__name__)
model_lasso = pickle.load(open("./model/Ai4i_lasso_final.pickle", "rb"))



@app.route('/')
@cross_origin()
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    processtemp = float(request.form.get('processtemp'))
    rotationalspeed = float(request.form.get('rotationalspeed'))
    torque = float(request.form.get("torquenm"))
    toolwear = float(request.form.get("toolwearmin"))
    twf = int(request.form.get('toolwearf'))
    hdf = int(request.form.get('heatfail'))
    pwf = int(request.form.get('powerfail'))
    osf = int(request.form.get('Verstrainfailure'))
    rnf = int(request.form.get('randomFail'))

    #print(processtemp, rotationalspeed, torque, toolwear, twf, hdf, pwf, osf, rnf)
    test = StandardScaler().fit_transform([[processtemp, rotationalspeed, torque, toolwear, twf, hdf, pwf, osf, rnf]])
    prediction = model_lasso.predict(test)[0]
    print(prediction)

    return str(np.round(prediction, 2))


@app.route('/profile', methods=['POST', 'GET'])
@cross_origin()
def profiling():
    return render_template("AI4I_Profiling.html")


if __name__ == '__main__':
    app.run()
