from flask import Flask
import pickle

app = Flask(__name__)

@app.route("/ping",methods=['GET'])
def ping():
    return {"msg ":" This your first demo Flask" }

@app.route("/harsh",methods=['GET'])
def hello_harsh():
    return "<p>Hello , Harsh</p>"