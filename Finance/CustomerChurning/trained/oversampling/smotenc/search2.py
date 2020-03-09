import numpy as np
import pandas as pd

#import needed functions
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

#import all the necessary models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from xgboost import XGBClassifier

# import the scoring/metrics functions we will use
# since it's classification
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import precision_score




from time import time



X_train = np.loadtxt('X_train')
X_test = np.loadtxt('X_test')
y_train = np.loadtxt('y_train')
y_test = np.loadtxt('y_test')


# create a nice dataframe to store results
results_df = pd.DataFrame(columns=['model', 'average_precision_score', 'balanced_accuracy_score',
	                                  'accuracy_score', 'recall_score', 'f1_score',
									                                    'cohen_kappa_score', 'precision_score', 'roc_auc_score'])



def choose_best(model, train_x , train_y , test_x , test_y):
    results = dict()
    results['model'] = model
    
    # for calculate time of fitting data
    start = time()
    model.fit(train_x,train_y)
    end = time()
    results['train_time'] = end-start
    
    # get the predicted results
    pred_y = model.predict(test_x)
    
    # calculate the various metrics
    results['average_precision_score'] = average_precision_score(test_y, pred_y)
    results['balanced_accuracy_score'] = balanced_accuracy_score(test_y, pred_y)
    results['accuracy_score'] = accuracy_score(test_y, pred_y)
    results['recall_score'] = recall_score(test_y, pred_y)
    results['f1_score'] = f1_score(test_y, pred_y)
    results['cohen_kappa_score'] = cohen_kappa_score(test_y, pred_y)
    results['precision_score'] = precision_score(test_y, pred_y)
    results['roc_auc_score'] = roc_auc_score(test_y, pred_y)
    return results






X_train = np.loadtxt('X_train')
X_test = np.loadtxt('X_test')
y_train = np.loadtxt('y_train')
y_test = np.loadtxt('y_test')



from imblearn.over_sampling import SMOTENC
smote_nc = SMOTENC(categorical_features=[4,5,6,7,8,9,10,11,12], random_state=0)
X_res, y_res = smote_nc.fit_resample(X_train, y_train)

geo_bool_mask = (X_res[:,6]+X_res[:,7]+X_res[:,8])==1
X_res = X_res[geo_bool_mask]
y_res = y_res[geo_bool_mask]

gender_bool_mask = (X_res[:,9]+X_res[:,10])==1
X_resampled = X_res[gender_bool_mask]
y_resampled = y_res[gender_bool_mask]





log_reg_param_grid = {'C':[0.01, 0.1, 1, 10, 100],
                     'max_iter':[100, 200, 300, 500],
                     'penalty':['l2'],
                     'class_weight':['balanced',None],
                     'tol': [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
                     }
log_reg = LogisticRegression()
log_reg_grid_search = GridSearchCV(log_reg, log_reg_param_grid,
                                  cv=5, refit=True, n_jobs=-1, scoring='average_precision')
log_reg_grid_search.fit(X_resampled,y_resampled)

# Run the model on the testing dataset to compute the various metrics
model = log_reg_grid_search.best_estimator_
results = choose_best(model, X_resampled, y_resampled, X_test, y_test)

# add the result to the big results dataframe
results_df = results_df.append(results, ignore_index=True)

# write the dataframe to a file
results_df.to_csv('oversampling_smotenc_log_reg.csv')

