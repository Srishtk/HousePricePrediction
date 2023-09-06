import numpy as np
import pickle
from flask import Flask, render_template, request, url_for

# flask app
app = Flask(__name__)

# get model
pickle_file = open('model.pkl', 'rb')
model = pickle.load(pickle_file)


# Home page
@app.route('/')
def home():
    return render_template('index.html')


@app.route("/predict", methods=["POST"])
def predict():
    area = float(request.form['area'])
    age = float(request.form['age'])
    bhk = float(request.form['bhk'])
    features = [area, age, bhk]
    # features_list = [features]

    predicted_val = model.predict([features])
    predicted_val = predicted_val[0]
    predicted_val = round(predicted_val, 2)
    predicted_val=str(predicted_val)
    return render_template("index.html", pred_text="The predicted value is {} INR.".format(predicted_val))


if __name__ == "__main__":
    app.run(debug=True)
