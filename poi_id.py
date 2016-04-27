#!/usr/bin/python

import sys
import pickle
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import RandomizedPCA
from sklearn.feature_selection import RFE
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

## Feature scaling using Min Max transformation
def scale_features(X):
    X = np.array(X)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    return X_scaled

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary', 'deferral_payments',
#        'total_payments', 'bonus', 'shared_receipt_with_poi',
#        'total_stock_value', 'expenses', 'from_this_person_to_poi', 
#        'deferred_income', 'long_term_incentive',
#        'from_poi_to_this_person', 'monetary_incentive', 'poi_contact'] # You will need to use more features

features_list = ['poi','salary', 'deferral_payments',
        'total_payments', 'bonus', 
        'total_stock_value', 
        'expenses', 
        'deferred_income', 
        'long_term_incentive',
        'monetary_incentive',
        #'salary', 'bonus', 'deferral_payments',
        'poi_contact'
        #'shared_receipt_with_poi', 'from_this_person_to_poi', 'from_poi_to_this_person'
        ] # You will need to use more features
        
### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
### Explore the dataset
    data_dict = pickle.load(data_file)

### Important features
total_no_of_dp = len(data_dict)
print "Number of records:", total_no_of_dp

data_dict_values = np.array(data_dict.values())
temp_list = list(filter(lambda d:d['poi'] == True, data_dict_values))

num_of_pois = len(temp_list)
print "Number of POI's:", num_of_pois
print "Number of non-POI's:", total_no_of_dp - num_of_pois

##############Start from here.. Try filtering values
for x in data_dict_values[1].keys():
    temp_list = list(filter(lambda d:d[x] == 'NaN', data_dict_values))
    print "The number of NaN values in ", x, " are ", len(temp_list)
#num_poi = len()

    
### Task 2: Remove outliers
### Pop out 'Total' from the data dictionary
data_dict.pop('TOTAL',0)

### Histogram plots for further exploration



### Task 3: Create new feature(s)
## Salary, bonus and deferral payments can be combined into a variable 
## 'monetary_incentive' that captures short term incentive of someone to 
## engage in illegal activities
## Similarly, 'shared_receipt_with_poi', 'from_this_person_to_poi' and 
## 'from_poi_to_this_person' is combined into a variable called 'poi_contact'
## It gives an indication of illegal activities too 
for key, value in data_dict.items():
    if value['salary'] == 'NaN':
        value['salary'] = 0
    if value['bonus'] == 'NaN':
        value['bonus'] = 0
    if value['deferral_payments'] == 'NaN':
        value['deferral_payments'] = 0
    if value['shared_receipt_with_poi'] == 'NaN':
        value['shared_receipt_with_poi'] = 0
    if value['from_this_person_to_poi'] == 'NaN':
        value['from_this_person_to_poi'] = 0
    if value['from_poi_to_this_person'] == 'NaN':
        value['from_poi_to_this_person'] = 0
    data_dict[key]['monetary_incentive'] = value['salary'] 
    + value['bonus'] + value['deferral_payments']
    data_dict[key]['poi_contact'] = value['shared_receipt_with_poi'] 
    + value['from_this_person_to_poi'] + value['from_poi_to_this_person']
        
### Store to my_dataset for easy export below.
temp_list_2 = list(filter(lambda d:d['restricted_stock_deferred'] != 'NaN', data_dict_values))
my_dataset = data_dict

#print features_list

### Extract features and labels from dataset for local testing
print features_list
data = featureFormat(my_dataset, features_list, sort_keys = True)
print data[1]
labels, features = targetFeatureSplit(data)

#print features

salary = []
deferral_payments = []

for point in data:
### Building numpy arrays from variables which seem unimportant
    salary.append(point[1])
    deferral_payments.append(point[2])

    
#n, bins, patches = plt.hist(salary, 50, facecolor='green', alpha=0.75)
plt.figure(1)
n, bins, patches = plt.hist(salary, 50, facecolor='green')
plt.xlabel('Salary')
plt.ylabel('Probability')
plt.title('Histogram of Salary')

plt.figure(2)
n, bins, patches = plt.hist(deferral_payments, 50, facecolor='green')
plt.xlabel('Deferral Payments')
plt.ylabel('Probability')
plt.title('Histogram of Deferral Payments')
plt.show()

### Scale the features
scaled_data = scale_features(data)
#print scaled_data[0]   


#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model, decomposition, datasets
from sklearn.pipeline import Pipeline
### A Naive Bayes classifier is fit to the data and its accuracy is tested
#clf = GaussianNB()
from sklearn.metrics import accuracy_score
#clf = clf.fit(features_train, labels_train)


#from sklearn import tree
#clf_DT = tree.DecisionTreeRegressor()

#from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.cross_validation import train_test_split
print features[1]
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Select Kbest features    
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2 
from sklearn.feature_selection import f_classif
features_train_ar = np.asarray(features_train)
features_test_ar = np.asarray(features_test)
print features_train_ar.shape
print features_train_ar.shape
labels_train_ar =  np.asarray(labels_train) 
labels_test_ar =  np.asarray(labels_test)
print labels_train_ar.shape
# features_train_new = SelectKBest(chi2, k=5).fit_transform(features_train_ar,labels_train_ar)
kbestclf = SelectKBest(f_classif, k=7).fit(features_train_ar,labels_train_ar)

# Get reduced feature set names
print kbestclf.get_support()
print kbestclf.scores_
features_train_new = kbestclf.transform(features_train_ar)
print features_train_new[1]

features_test_new = kbestclf.transform(features_test_ar)

from sklearn.decomposition import PCA
# Apply PCA and Naive Bayes on the new feature set
pca = decomposition.PCA()
clf = Pipeline(steps=[('pca', pca), ('gaussian_NB', GaussianNB())])
n_components = [5, 6, 7]
clf = GridSearchCV(clf, dict(pca__n_components=n_components))
clf = clf.fit(features_train_new, labels_train_ar)
print "The number of components of the best estimator is ", clf.best_estimator_.named_steps['pca'].n_components

## Apply Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf = clf.fit(features_train_new, labels_train)


# Apply Decision Tree
#from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier(random_state=0, min_samples_split=5, max_features=4)
#clf = DecisionTreeClassifier(random_state=0, min_samples_split=5, max_features=7)
#clf = clf.fit(features_train_new, labels_train)




#features_test_new = kbestclf.transform(features_test)
#features_pred = clf.predict(features_test_new) 
#features_pred = clf.predict(features_test) 
    
### A Naive Bayes classifier combined with PCA is used and its accuracy is tested
#from sklearn.decomposition import PCA
#pca = decomposition.PCA()
#pca_clf = pca.fit(features_train_ar, labels_train)




#print "Explained Variance Ratio: ", pca_clf.explained_variance_ratio_
#print "Components: ", pca_clf.components_
#features_train_new = pca_clf.transform(features_train_ar)

## Apply Naive Bayes
#from sklearn.naive_bayes import GaussianNB
#clf = GaussianNB()
#clf = clf.fit(features_train_new, labels_train)




#clf = GaussianNB()
#clf = Pipeline(steps=[('pca', pca), ('gaussian_NB', GaussianNB())])
#n_components = [3, 5, 7, 9]
#clf = GridSearchCV(clf,
#                         dict(pca__n_components=n_components))
# from sklearn.tree import DecisionTreeClassifier
#clf = DecisionTreeClassifier(random_state=0, min_samples_split=20)
#clf = clf.fit(features_train, labels_train)
#print "The feature mask is:", clf.get_support()
#features_pred = clf.predict(features_test) 
# print "The number of components of the best estimator is ", clf.best_estimator_.named_steps['pca'].n_components
#print "The best parameters:", clf.best_params_
#print "The best estimator", clf.best_estimator_.get_params(deep=True).gaussian_NB
# best_est = RFE(clf.best_estimator_)
# print "The best estimator:", best_est
#estimator = clf
#estimator = clf.named_steps['features'].get_feature_names()
#print "The features are:", estimator
#['features'].get_feature_names()
# print estimator.named_steps['gaussian_NB'].get_support()
#['features'].get_feature_names()
#features_pred = clf.predict(features_test)    
    
### Feature selection using Decision Trees
#clf_feature_sel_DT = ExtraTreesClassifier()
#clf_feature_sel_DT = clf_feature_sel_DT.fit(features_train, labels_train)
#print "The old feature shape is", features_train.shape
#feature_DT_model = SelectFromModel(clf_feature_sel_DT, prefit=True)
#feature_DT_train = feature_DT_model.transform(features_train)
#print "The new feature shape using DT is", feature_DT_train.shape

### Feature selection algorithm using PCA
#n_components = 250
#pca = RandomizedPCA(n_components=n_components, whiten=True).fit(features_train)
#features_train_pca =  pca.transform(features_train)
#features_test_pca = pca.transform(features_test)
#print len(features_train[0])
#print len(features_train_pca[0])
    

#print "The accuracy score of the Naive Bayes Classifier is: " , accuracy_score(labels_test, features_pred)

#clf_NB = clf_NB.fit(features_train_pca, labels_train)
#features_pred_NB = clf_NB.predict(features_test_pca)
#print "The accuracy score of the Naive Bayes Classifier using PCA is: " , accuracy_score(labels_test, features_pred_NB)

### A decision tree is used
#clf_DT = clf_DT.fit(features_train, labels_train)
#features_pred_DT = clf_DT.predict(features_test)
#print "The accuracy score of the Decision Tree is: " , accuracy_score(labels_test, features_pred_DT)
    
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)