import pickle
import website.models.clean as clean
import pandas as pd

from pymongo import MongoClient

## Convert dictionary into pandas frame
def make_pandas(entry):

	prev = entry['previous_payouts']
	## Adding column with length of previous payouts
	entry['dict_elements'] = len(prev)
	del entry['previous_payouts']

	df = pd.DataFrame.from_dict(entry)

	## Clean the data
	return clean.clean_data_new(df)

def flags(raw_data, clas):
	flags = []
	if raw_data['currency'] not in ['USD', 'CAD', 'GBP', 'AUD', 'EUR', 'NZD']:
		flags.append('Unusual Currency')
	if raw_data['user_age'] < 10:
		flags.append('New User')
	if raw_data['payoutdiff'] > 10000000:
		flags.append('High payoutdiff')
	if raw_data['gts'] > 1500:
		flags.append('High gts')
	if raw_data['num_order'] < 2:
		flags.append('Low num_order')
	if raw_data['payee_exists'] != True:
		flags.append('No valid payee')
	if raw_data['dict_elements'] < 2:
		flags.append('Sparse payment history')

	return flags


## Predict on the new entry
def predict(model, cleaned, cols):

	## Selecting on important columns and getting preds
	X = cleaned[cols].values
	preds = model.predict_proba(X)[:,1]

	return preds


def get_prediction(d):

	## Read in the pickle model
	model = pickle.load(open('website/models/rf_model.p', 'rb'))

	# ## Read in the first X new entries
	# r = table.find().sort([('_id', -1)]).limit(2)

	## Transforming data
	cleaned = make_pandas(d)
	

	## Predict on the new data
	rf_cols = ['USD','GBP','CAD','AUD','EUR','NZD','MXN', 
		   'age_dummy',
		   'user_age',
		   'payoutdiff',
		   'gts',
		   'num_order',
		   'num_payouts',
		   'payee_exists',
		   'dict_elements']
	return predict(model, cleaned, rf_cols)
	

