'''
Live Data File is used to:
	1. Read in new event entry from website and put into MongoDB
	2. Predict fraud probability and asign it a classification
	3. Update data with new pred info into MongoDB
'''

from pymongo import MongoClient
import requests
import time
import numpy as np

from website.models import predict

client = MongoClient('localhost', 27017)
db = client['fraud']
table = db['events']


api_key = 'vYm9mTUuspeyAWH1v-acfoTlck-tCxwTw9YfCynC'

url = 'http://galvanize-case-study-on-fraud.herokuapp.com/data_point'
sequence_number = 0


response = requests.get(url)
raw_data = response.json()


def is_new(raw_data):

	new_id = raw_data['object_id']
	if len(list(table.find({'object_id': new_id}))) > 0:
		return False

	else:
		return True


while True:
	try:

		## Reading in JSON
		response = requests.get(url)
		raw_data = response.json()


		new = is_new(raw_data)

		if new:
			print("New Entry")

			## Getting prediction form predictions file
			pred = np.mean(predict.get_prediction(raw_data))

			## Assignning and class
			clas = 'Low'
			if pred >= .5:
				clas = 'Medium'
			elif pred > .75:
				clas = 'High'

			#fs = flags(raw_data, clas)

			print("Prediction:", pred, "  Class:", clas)

			## Inserting prediction and class into dictionary
			raw_data['prediction'] = pred
			raw_data['risk'] = clas
			#raw_data['flags'] = fs

			## Insert raw data
			table.insert_one(raw_data)

			sequence_number += 1
			print(sequence_number)

		print("Duplicate")

	except:
		print('Failed to Predict')
			
		## Sleep program for 1.5 minutes, waiting for new data
	time.sleep(10)
