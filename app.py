from flask import Flask, render_template, request
import numpy as np
import pickle

# Load models
heart_model = pickle.load(open('models/heart.pkl', 'rb'))
liver_model = pickle.load(open('models/liver.pkl', 'rb'))
kidney_model = pickle.load(open('models/kidney.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/heart", methods=['GET', 'POST'])
def heart():
    return render_template('heart.html')

@app.route("/kidney", methods=['GET', 'POST'])
def kidney():
    return render_template('kidney.html')

@app.route("/liver", methods=['GET', 'POST'])
def liver():
    return render_template('liver.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Heart Disease Prediction
        if len([float(x) for x in request.form.values()]) == 13:
            age = int(request.form['age'])
            sex = int(request.form['sex'])
            cp = int(request.form['cp'])
            trestbps = int(request.form['trestbps'])
            chol = int(request.form['chol'])
            fbs = int(request.form['fbs'])
            restecg = int(request.form['restecg'])
            thalach = int(request.form['thalach'])
            exang = int(request.form['exang'])
            oldpeak = float(request.form['oldpeak'])
            slope = int(request.form['slope'])
            ca = int(request.form['ca'])
            thal = int(request.form['thal'])

            data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            data1 = np.array(data).reshape(1, -1)
            my_prediction = heart_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)

        # Liver Disease Prediction
        elif len([float(x) for x in request.form.values()]) == 10:
            Age = int(request.form['Age'])
            Total_Bilirubin = float(request.form['Total_Bilirubin'])
            Direct_Bilirubin = float(request.form['Direct_Bilirubin'])
            Alkaline_Phosphotase = int(request.form['Alkaline_Phosphotase'])
            Alamine_Aminotransferase = int(request.form['Alamine_Aminotransferase'])
            Aspartate_Aminotransferase = int(request.form['Aspartate_Aminotransferase'])
            Total_Protiens = float(request.form['Total_Protiens'])
            Albumin = float(request.form['Albumin'])
            Albumin_and_Globulin_Ratio = float(request.form['Albumin_and_Globulin_Ratio'])
            Gender_Male = int(request.form['Gender_Male'])

            data = np.array([[Age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase,
                              Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens,
                              Albumin, Albumin_and_Globulin_Ratio, Gender_Male]])
            my_prediction = liver_model.predict(data)
            return render_template('predict.html', prediction=my_prediction)

        # Kidney Disease Prediction
        elif len([float(x) for x in request.form.values()]) == 18:
            age = float(int(request.form['age']))
            bp = float(request.form['bp'])
            al = float(request.form['al'])
            su = float(request.form['su'])
            rbc = int(request.form['rbc'])
            pc = int(request.form['pc'])
            pcc = int(request.form['pcc'])
            ba = int(request.form['ba'])
            bgr = float(request.form['bgr'])
            bu = float(request.form['bu'])
            sc = float(request.form['sc'])
            pot = float(request.form['pot'])
            wc = int(request.form['wc'])
            htn = int(request.form['htn'])
            dm = int(request.form['dm'])
            cad = int(request.form['cad'])
            pe = int(request.form['pe'])
            ane = int(request.form['ane'])

            data = [age, bp, al, su, rbc, pc, pcc, ba, bgr, bu, sc, pot, wc, htn, dm, cad, pe, ane]
            data1 = np.array(data).reshape(1, -1)
            my_prediction = kidney_model.predict(data1)
            return render_template('predict.html', prediction=my_prediction)

if __name__ == "__main__":
    app.run(debug=True)
