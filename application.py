from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model=pickle.load(open("quikr_car.pkl",'rb'))
@app.route('/')
def index():
    return render_template('index.html')  # no slash before index.html

@app.route('/predict', methods=['POST'])
def predict():
    present_price = request.form.get('cp')
    age = request.form.get('age')
    dist = request.form.get('dist')
    fuel_type = request.form.get('fuel')

    print(present_price, age, dist, fuel_type)

    prediction=model.predict(np.array([present_price,dist,age,fuel_type]).reshape(1, -1))
    # # Example prediction logic (replace with your model if needed)
    # try:
    #     price = float(present_price) - (float(age) * 1000) - (float(dist) * 0.05)
    #     if int(fuel_type) == 2:
    #         price -= 500
    #     price = max(0, round(price, 2))  # Ensure price isn't negative
    # except:
    #     price = "Invalid input"
    print(prediction)
    return str(prediction[0])  # Return prediction as plain text

if __name__ == "__main__":
    app.run(debug=True)
