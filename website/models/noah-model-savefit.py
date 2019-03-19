import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import clean	## Importing cleaning file locally

## To upsumple no fraud class
from imblearn.over_sampling import SMOTE

import warnings
warnings.simplefilter('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV



######################################################
###### SCORING METRICS and PROFIT CURVES #############

## Gets recall and auc score of model
def cross_val_recall_auc(X_train, y_train, func):

	rec = sum(cross_val_score(func, X_train, y_train, scoring='recall'))/3
	auc = sum(cross_val_score(func, X_train, y_train, scoring='roc_auc'))/3

	func_name = str(func.__class__.__name__)

	print("{0:27} Train CV | Recall: {1:5.4} | AUC: {2:5.4}".format(func_name, rec, auc))
	return rec, auc

## Gets the confusion matrix and scores for prediction and target
def get_confusion(preds, target):
	TP = (preds & target).sum()
	FP = ((preds - target) == 1).sum()
	TN = ((preds + target) == 0).sum()
	FN = ((preds - target) == -1).sum()

	confusion = [TP, FP,
				 FN, TN]

	actual_p = target.sum()
	actual_n = len(target) - actual_p

	scoring = [TP/actual_p, FP/actual_n,
			   FN/actual_p, TN/actual_n ]


	cons = ['TP', 'FP', 'FN', 'TN']
	scos = ['TPR', 'FPR', 'FNR', 'TNR']
	for i in range(4):
		print("{0:2}: {1:5}  |  {2:3}: {3:5.3}".format(cons[i],confusion[i],scos[i],scoring[i]))

	return (confusion, scoring)

## Find optimal threshold
def find_thresh(preds, y_test, samples):
	thresholds = np.linspace(0, .65, samples)
	for i in thresholds:
		print('\nThreshold:',np.round(i, 4))
		new = (preds > i).astype(int)
		(conf, score) = get_confusion(new, y_test)

## Creates confusion matrix from true and predicted np arrays
def standard_confusion_matrix(y_true, y_predict):
	y_true, y_predict = y_true>0, y_predict>0

	tp, fp = (y_true & y_predict).sum(), (~y_true & y_predict).sum()
	fn, tn = (y_true & ~y_predict).sum(), (~y_true & ~y_predict).sum()
	return np.array([[tp, fp], [fn, tn]])

## Returns dictionary of threshold and assocated profit val for given cost_ben
def profit_curve(cost_ben, pred_probs, labels):
	order = pred_probs.argsort()[::0-1]	## list of indexes to sort in reverse order
	thresholds, profits = [], []

	for ind in order:
		thresh = pred_probs[ind]	## Sets theshold to prob
		pos_class = pred_probs > thresh
		confusion_mat = standard_confusion_matrix(labels, pos_class)
		profit = (confusion_mat * cost_ben).reshape(1,-1).sum()/len(labels)
		
		profits.append(profit)
		thresholds.append(thresh)

	plt.plot(thresholds, profits)
	plt.title("Profit Curve")
	plt.xlabel("Percentage of test instances (decreasing by score)")
	plt.ylabel("Profit")
	plt.grid()
	plt.savefig('website/images/profit_curve.png')
	plt.show()
	
	return (thresholds, profits)

######################################################



######################################################
################## MODELS ############################

## Logistic regression model with 
def fit_logreg(y, cleaned, cols):

	X = cleaned[cols].values

	X_train, X_test, y_train, y_test = train_test_split(X,
														y,
														test_size=.2,
														random_state=1)

	## Resampling the data to avoid non fraud bias
	method = SMOTE(kind='regular')
	X_train, y_train = method.fit_sample(X_train, y_train)


	model = LogisticRegression()
	model.fit(X_train, y_train)
	preds = model.predict_proba(X_test)[:,1]
	print(preds)

	find_thresh(preds, y_test, 10)

	cross_val_recall_auc(X_train, y_train, model)

## Random Forrest Model
def fit_rf(y, cleaned, cols):
	X = cleaned[cols].values

	X_train, X_test, y_train, y_test = train_test_split(X,
														y,
														test_size=.2,
														random_state=1)

	## Resampling the data to avoid non fraud bias
	method = SMOTE(kind='regular')
	X_train, y_train = method.fit_sample(X_train, y_train)

	# rf_grid(X_train, y_train, X_test, y_test)

	model = RandomForestClassifier(bootstrap= True,
								   max_depth= 3,
								   max_features= 'sqrt',
								   min_samples_leaf= 4,
								   min_samples_split= 2,
								   n_estimators= 20,
								   random_state= 1)
	model.fit(X_train, y_train)
	pickle.dump(model, open('website/models/rf_model.p', 'wb'))

	preds = model.predict_proba(X_test)[:,1]
	print(preds)

	cost_ben = np.array([[ 4990, -450],
						 [ 0, 0]])
	profit_curve(cost_ben, preds, y_test)

	find_thresh(preds, y_test, 10)

	cross_val_recall_auc(X_train, y_train, model)
	return model

## Gradient Boost Model
def fit_gb(y, cleaned, cols):
	X = cleaned[cols].values


	X_train, X_test, y_train, y_test = train_test_split(X,
														y,
														test_size=.2,
														random_state=1)

	## Resampling the data to avoid non fraud bias
	method = SMOTE(kind='regular')
	X_train, y_train = method.fit_sample(X_train, y_train)


	model = GradientBoostingClassifier(max_depth= 3,
									   max_features= 'sqrt',
									   min_samples_leaf= 4,
									   min_samples_split= 2,
									   n_estimators= 5,
									   random_state= 1)
	model.fit(X_train, y_train)
	pickle.dump(model, open('website/models/gb_model.p', 'wb'))

	preds = model.predict_proba(X_test)[:,1]
	print(preds)

	find_thresh(preds, y_test, 10)

	cross_val_recall_auc(X_train, y_train, model)

## Grid Search for the Random Forrest
def rf_grid(X_train, y_train, X_test, y_test):
	random_forest_grid = {'max_depth': [3, 5, 7],
					  'max_features': ['sqrt', 'log2'],
					  'min_samples_split': [2, 4],
					  'min_samples_leaf': [1, 2, 4],
					  'bootstrap': [True, False],
					  'n_estimators': [5, 10, 20],
					  'random_state': [1]}

	rf_gridsearch = GridSearchCV(RandomForestClassifier(),
								 random_forest_grid,
								 n_jobs=-1,
								 verbose=True,
								 scoring='recall')
	rf_gridsearch.fit(X_train, y_train)

	print( "best parameters:", rf_gridsearch.best_params_ )

	best_rf_model = rf_gridsearch.best_estimator_
	preds = best_rf_model.predict(X_test)

	find_thresh(preds, y_test, 10)

	cross_val_recall_auc(X_train, y_train, best_rf_model)

####################################################


###############################################
############### MAIN ##########################

if __name__ == '__main__':
	df = pd.read_json('website/data/data.json')

	## Clean the data
	cleaned = clean.clean_data(df)
	print(cleaned.columns)

	## Getting targets and cleaned features
	y = clean.get_target(cleaned)

	# log_reg_cols = ['user_age', 'age_dummy']
	# fit_logreg(y, cleaned, log_reg_cols)


	#Cols: 'USD','GBP','CAD','AUD','EUR','NZD','MXN','age_dummy','user_age','payoutdiff', 'eventdiff', 0.0, 1.0, 3.0, 'gts', 'num_order', 'num_payouts','payee_exists'
	rf_cols = ['USD','GBP','CAD','AUD','EUR','NZD','MXN', 
			   'age_dummy',
			   'user_age',
			   'payoutdiff',
			   'gts',
			   'num_order',
			   'num_payouts',
			   'payee_exists',
			   'dict_elements']
			   
	rf_model = fit_rf(y, cleaned, rf_cols)

	gb_model = fit_gb(y, cleaned, rf_cols)



	



