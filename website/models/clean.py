import pandas as pd
import numpy as np


######################################################
################## CLEANING ##########################

def clean_data(df):

	## Age Dummy if over specified number
	df['age_dummy'] = df['user_age'].apply(lambda x: 1 if x > 0 else 0)

	## Adding Payoutdate and Eventdate Diff columns
	df['eventdiff'] = df['event_published'] - df['event_end']
	df['payoutdiff'] = df['approx_payout_date'] - df['event_created']

	## Getting dummies on curreny and delivery method
	df = pd.concat([df, pd.get_dummies(df.currency)], axis=1);
	df = pd.concat([df, pd.get_dummies(df.delivery_method)], axis=1);

	## Stripping the 
	df['payee_exists'] = [x.strip()=="" for x in df['payee_name']]

	## Adding column with length of previous payouts
	df['dict_elements'] = df.previous_payouts.map(lambda x: len(x))

	return df

def clean_data_new(df):
	## Age Dummy if over specified number
	df['age_dummy'] = df['user_age'].apply(lambda x: 1 if x > 0 else 0)

	## Adding Payoutdate and Eventdate Diff columns
	df['eventdiff'] = df['event_published'] - df['event_end']
	df['payoutdiff'] = df['approx_payout_date'] - df['event_created']

	## Getting dummies on curreny and delivery method
	df = pd.concat([df, pd.get_dummies(df.currency)], axis=1);
	df = pd.concat([df, pd.get_dummies(df.delivery_method)], axis=1); 

	## Stripping the 
	df['payee_exists'] = [x.strip()=="" for x in df['payee_name']]

	## Cleaning data for one entry at a time
	df['AUD'] = (df['currency']=='AUD').astype(int)
	df['CAD'] = (df['currency']=='CAD').astype(int)
	df['EUR'] = (df['currency']=='EUR').astype(int)
	df['GBP'] = (df['currency']=='GBP').astype(int)
	df['MXN'] = (df['currency']=='MXN').astype(int)
	df['NZD'] = (df['currency']=='NZD').astype(int)
	df['USD'] = (df['currency']=='USD').astype(int)

	df[0.0] = (df['delivery_method']==0.0).astype(int)
	df[1.0] = (df['delivery_method']==1.0).astype(int)
	df[3.0] = (df['delivery_method']==3.0).astype(int)

	return df


def derek_clean(df):
	df['facebook_presence'] = df.org_facebook.apply(lambda x:1 if x>5 else 0)
	df['twitter_presence'] = df.org_twitter.apply(lambda x:1 if x>5 else 0)
	df['have_previous_payouts'] = df['previous_payouts'].apply(lambda x: 1 if len(x) != 0 else 0)
	df['highly_suspect_state'] = df['venue_state'].apply(lambda x: 1 if x in ['MT', 'Mt', 'AK', 'FL', 'NEW SOUTH WALES', 'Florida'] else 0)
	df['cap_name'] = df['name'].apply(lambda x: 1 if x.isupper() == True else 0)
	df['org_desc_len'] = [len(x) for x in df['org_desc']]
	df['payee_in_org'] = df.apply(lambda x: x.payee_name in x.org_name, axis=1)
	previous_payouts = df['previous_payouts']
	df['useage']=[len(x)for x in previous_payouts]
	df['useage_bool'] = df.useage.apply(lambda x:0 if x>1 else 1)

	df['AUD'] = df['currency']=='AUD'
	df['CAD'] = df['currency']=='CAD'
	df['EUR'] = df['currency']=='EUR'
	df['GBP'] = df['currency']=='GBP'
	df['MXN'] = df['currency']=='MXN'
	df['NZD'] = df['currency']=='NZD'
	df['USD'] = df['currency']=='USD'

	df[0.0] = df['delivery_method']==0.0
	df[1.0] = df['delivery_method']==1.0
	df[3.0] = df['delivery_method']==3.0

## Cleaning the target data into 1 and 0
def get_target(df):

	## Signals fraud account 
	fraud_accts = set(['fraudster_event', 'fraudster', 'fraudster_att'])

	new_df = df.copy()
	new_df['fraud'] = df['acct_type'].apply(lambda x: 1 if x in fraud_accts else 0)
	new_df.drop('acct_type', axis=1, inplace=True)	## Dropping old col
	return new_df['fraud'].values