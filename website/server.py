## Server Flask File

from flask import Flask, render_template, request, jsonify, Response
import pickle
import numpy as np
import pandas as pd

from pymongo import MongoClient
client = MongoClient('localhost', 27017)
db = client['fraud']
table = db['events']

## Create the app object that will route our calls
app = Flask(__name__)


## Rendering the home page HTML
@app.route('/', methods = ['GET'])
def home():
	r = table.find().sort([('_id', -1)]).limit(50)

	items = []
	for entry in r:
	    items.append([entry['object_id'],
	    			  entry['risk'],
	    			  np.round(entry['prediction'], 3)])


	return render_template('home.html', data=items)


@app.route('/info', methods = ['GET'])
def info():
	return render_template('info.html')


## Calculating and posting the linear regression model prediction
@app.route('/prediction', methods = ['POST', 'GET'])
def prediction():

	if request.method == 'POST':

		r = table.find().sort([('_id', -1)]).limit(50)

		items = []
		for entry in r:
		    items.append([entry['object_id'],
		    			  entry['risk'],
		    			  entry['prediction']])


		return render_template('home.html', data=items)
	## Returning json formatted output (.js file grabs 'prediction')
	#return jsonify({'prediction':np.round(pred[0],3)})


if __name__ == '__main__':
	app.run(host='0.0.0.0', port=3333, debug=True)

