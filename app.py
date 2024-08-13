import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("housing-prices-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    number_rooms = int(request.form['bedroom_count'])
    distance_city = float(request.form['center_distance'])
    distance_metro = float(request.form['metro_distance'])
    age_house = int(request.form['age'])
    net_area = float(request.form['net_sqm'])

    
    final_features = np.array([[number_rooms, net_area, distance_city, distance_metro, age_house]])
    prediction = model.predict(final_features)




    return render_template(
        "index.html", prediction_text="{}".format(prediction)
    )


if __name__ == "__main__":
    app.run()
