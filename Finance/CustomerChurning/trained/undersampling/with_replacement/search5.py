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



from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(random_state=0, replacement=True)
X_resampled, y_resampled = rus.fit_sample(X_train, y_train)








# Time for XGBoost
xgb_param_grid = {'max_depth':[3,4,5,6,7,8],
        'gamma':[1e-2, 1e-3, 1e-4],
        'min_child_weight':[1,5,10,20],
        'n_estimators':[1,5,10,50,100],
        'min_child_weight':[1,10,20]
        }
xgb_clas = XGBClassifier()
xgb_grid_search = GridSearchCV(xgb_clas, xgb_param_grid,
        cv = 5,
        refit = True,
        n_jobs = -1,
        scoring = 'average_precision')
xgb_grid_search.fit(X_resampled, y_resampled)
# Run the model on the testing dataset to compute the various metrics
model = xgb_grid_search.best_estimator_
results = choose_best(model, X_resampled, y_resampled, X_test, y_test)

# add the result to the big results dataframe
results_df = results_df.append(results, ignore_index=True)


# write the dataframe to a file
results_df.to_csv('undersampling_with_replacement_xgb.csv')

