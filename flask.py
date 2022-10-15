from flask import Flask, jsonify, request
#Importing the function from the classifier.py file
from classifier import getPrediction

app = Flask(__name__)

@app.route("/predict-alphabet", methods = ["POST"])
def predictData():
    image = request.files.get("alphabet")
    #predicting using the getPrediction function
    prediction = getPrediction(image)
    return jsonify({
        "prediction" : prediction
    }), 200
    #Above, 200 is the success code
    
if __name__ == "__main__":
    app.run(debug = True)        