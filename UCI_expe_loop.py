
from __future__ import print_function
import numpy as np
from sklearn.linear_model import Ridge, lars_path
from sklearn.utils import check_random_state
from synthetic_datasets import *

import pandas, numpy, sklearn
from local_surrogate_analysis import *
from sklearn.metrics import pairwise_distances
import load_dataset
import sklearn.model_selection, sklearn.ensemble
import ugs_new_cap as gs
from sklearn import linear_model
import lime_assessment
from sklearn.metrics import roc_auc_score



def accuracy(model_lime_pred, dataset, pred_fn, measure, label_toexplain):
    y_clf = pred_fn(dataset)[:,label_toexplain]
    y_pred_lime = model_lime_pred(dataset)
    return measure(y_clf, y_pred_lime) 

def lime_pred(obs, exp, label_toexplain, dataset_ref, regression): #entrée observation, sortie pred
    dataset_ref = (dataset_ref-exp.mean_)/exp.scale_
    surrogate = exp.easy_model[label_toexplain]
    y_m_surrogate_pred_proba =  surrogate.predict(obs.iloc[:,exp.used_features[label_toexplain]])
    if regression:
        return y_m_surrogate_pred_proba
    else:
        # OK for a binary classification problem
        y_m_surrogate_pred = numpy.zeros(obs.shape[0])
        mask = y_m_surrogate_pred_proba<0.5
        y_m_surrogate_pred[mask] = 1-label_toexplain
        mask = y_m_surrogate_pred_proba>=0.5
        y_m_surrogate_pred[mask] = label_toexplain
        return y_m_surrogate_pred

def lime_local_results(x_toexplain, dataset_train_lime, dataset_accuracy, pred_fn, label_toexplain, num_features, kernel_width=None):
    #entrée observation, dataset train lime, dataset accuracy, retour accuracy
    explainer = lime_assessment.lime_tabular.LimeTabularExplainer(dataset_train_lime, 
                                                                  discretize_continuous=False, kernel_width=kernel_width)
    exp = explainer.explain_instance(x_toexplain,
                                     pred_fn,
                                     num_features=num_features,
                                     top_labels=len(ylabels), 
                                     labels=range(len(ylabels)))
    
    
    regression = True
    model_lime_pred = lambda obs: lime_pred(obs, exp, label_toexplain, dataset_train_lime, regression=regression)
    if regression == False:
        pred_function = lambda obs: pred_fn(obs)>=0.5
        measure = accuracy_score
    else:
        pred_function = lambda obs: pred_fn(obs)>=0.5
        measure = roc_auc_score#r2_score
    acc = accuracy(model_lime_pred, dataset_accuracy, pred_function, measure, label_toexplain)
    
    '''
    ici on va sortir stab = stability()
    '''
    return acc

def prop_local_results(x_toexplain, dataset_train_lime, dataset_accuracy, pred_fn, label_toexplain, num_features, radius_train=0.3):
    #entrée observation, dataset train lime, dataset accuracy, retour accuracy
    #pred = lambda x: int(pred_fn(x)>=0.5)
    closest_enn, _ = gs.main(clf.predict, x_toexplain.reshape(1,-1), n_layer=1000, first_radius=0.1, step_size=100)
    closest_dist = float(pairwise_distances(x_toexplain.reshape(1,-1), closest_enn.reshape(1, -1))[0])
    X_local = generate_inside_ball(closest_enn.reshape(1,- 1), segment=(0, radius_train), n=1000)
    y_local = pred_fn(X_local)[:, 1]

    
    
    '''local_lr = linear_model.LinearRegression()
    local_lr = local_lr.fit(X_local, y_local)
    model_prop_pred = local_lr.predict'''
    weights = np.ones(X_local.shape[0])
    used_features = feature_selection(X_local, y_local, weights, num_features, 'lasso_path')

    model_regressor = Ridge(alpha=1, fit_intercept=True,
                                    random_state=0)
    easy_model = model_regressor
    easy_model.fit(X_local[:, used_features],
                       y_local, sample_weight=weights)
    model_prop_pred = lambda obs: easy_model.predict(obs.iloc[:, used_features])
    
    
    regression = True
    if regression == False:
        pred_function = lambda obs: pred_fn(obs)>=0.5
        measure = accuracy_score
    else:
        pred_function = lambda obs: pred_fn(obs)>=0.5
        measure = roc_auc_score#r2_score
    acc = accuracy(model_prop_pred, dataset_accuracy, pred_function, measure, label_toexplain)
    
    '''
    ici on va sortir stab = stability()
    '''
    return acc

def get_random_points_within_hypersphere(x_toexplain, r=1, N=100):
    # Generate N random points in a hypersphere of radius r
    res = []
    N_todraw = N
    while len(res) < N:
        N_todraw = N - len(res)
        X_generated = numpy.random.uniform(low=[x_toexplain-r for _ in range(N_todraw)], high=[x_toexplain+r for _ in range(N_todraw)])
        dists = euclidean_distances(x_toexplain.to_frame().T, X_generated)[0]
        for i in range(X_generated.shape[0]):
            if dists[i] < r:
                res.append(pandas.Series(X_generated[i], x_toexplain.index))
    X_generated = pandas.DataFrame(res)
    return X_generated








def generate_lars_path(weighted_data, weighted_labels):
        """Generates the lars path for weighted data.

        Args:
            weighted_data: data that has been weighted by kernel
            weighted_label: labels, weighted by kernel

        Returns:
            (alphas, coefs), both are arrays corresponding to the
            regularization parameter and coefficients, respectively
        """
        x_vector = weighted_data
        alphas, _, coefs = lars_path(x_vector,
                                     weighted_labels,
                                     method='lasso',
                                     verbose=False)
        return alphas, coefs

def forward_selection(data, labels, weights, num_features):
        """Iteratively adds features to the model"""
        clf = Ridge(alpha=0, fit_intercept=True, random_state=0)
        used_features = []
        for _ in range(min(num_features, data.shape[1])):
            max_ = -100000000
            best = 0
            for feature in range(data.shape[1]):
                if feature in used_features:
                    continue
                clf.fit(data[:, used_features + [feature]], labels,
                        sample_weight=weights)
                score = clf.score(data[:, used_features + [feature]],
                                  labels,
                                  sample_weight=weights)
                if score > max_:
                    best = feature
                    max_ = score
            used_features.append(best)
        return np.array(used_features)
    
def feature_selection(data, labels, weights, num_features, method='lasso_path'):
        """Selects features for the model. see explain_instance_with_data to
           understand the parameters."""
        if method == 'none':
            return np.array(range(data.shape[1]))
        elif method == 'forward_selection':
            return forward_selection(data, labels, weights, num_features)
        elif method == 'highest_weights':
            clf = Ridge(alpha=0, fit_intercept=True,
                        random_state=0)
            clf.fit(data, labels, sample_weight=weights)
            feature_weights = sorted(zip(range(data.shape[0]),
                                         clf.coef_ * data[0]),
                                     key=lambda x: np.abs(x[1]),
                                     reverse=True)
            return np.array([x[0] for x in feature_weights[:num_features]])
        elif method == 'lasso_path':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_labels = ((labels - np.average(labels, weights=weights))
                               * np.sqrt(weights))
            nonzero = range(weighted_data.shape[1])
            _, coefs = generate_lars_path(weighted_data,
                                               weighted_labels)
            for i in range(len(coefs.T) - 1, 0, -1):
                nonzero = coefs.T[i].nonzero()[0]
                if len(nonzero) <= num_features:
                    break
            used_features = nonzero
            return used_features
        elif method == 'auto':
            if num_features <= 6:
                n_method = 'forward_selection'
            else:
                n_method = 'highest_weights'
            return feature_selection(data, labels, weights,
                                          num_features, n_method)



def generate_inside_ball(center, segment=(0,1), n=1):
    def norm(v):
        return np.linalg.norm(v, ord=2, axis=1)
    d = center.shape[1]
    z = np.random.normal(0, 1, (n, d))
    z = np.array([a * b / c for a, b, c in zip(z, np.random.uniform(*segment, n),  norm(z))])
    z = z + center
    return z # les z sont a distance de center comprise dans le segment 

ylabels = ['Class 0', 'Class 1']

DATASETS = ['news']#['news']#['cancer', 'credit']

output = {}

for d_name in DATASETS:
    print("================== Working on dataset", d_name, "==================")
    X, y = get_moons(n_samples=1000, random_state=4)
    ylabels = ['Class 0', 'Class 1']
    feature_names = ['feature 0', 'feature 1']
    '''X, y = load_dataset.main(d_name, n_obs=1000)
    X = pandas.DataFrame(X)
    y = pandas.Series(y)
    y = y[(X.abs()>3).sum(axis=1)==0]
    X = X[(X.abs()>3).sum(axis=1)==0]'''
    print(X.shape)
    output[d_name] = (-1, -1, -1)

    NUM_FEATURES = X.shape[1]

    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(X, y, test_size=0.10)

    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=200)
    clf.fit(train, labels_train)
    print(sklearn.metrics.accuracy_score(labels_test, clf.predict(test)))
    
    dataset = test
    RADIUS_PERC = 0.2   #numpy.arange(2,11)/10.
    KW = 0.5#(0.75 * (X.shape[1])**(0.5)) / 2 #new kernel to try

    missing = 0
    global_lime_i = []
    kernel_lime_i = []
    prop_i = []
    for i in range(test.shape[0]):
        print("======",i,"====== on ", test.shape[0])
        x_toexplain = test.iloc[i,:]
        dists = euclidean_distances(x_toexplain.to_frame().T, dataset)
        dists = pandas.Series(dists[0], index=dataset.index)
        radius = RADIUS_PERC*dists.max()

        print('Radius used for accuracy', radius)
        X_t = pandas.DataFrame(generate_inside_ball(np.array(x_toexplain).reshape(1,-1), segment=(0,radius), n=1000))#get_random_points_within_hypersphere(x_toexplain, r, N=1000)

        #X_t = dataset.loc[dists[dists<=r].index]


        #for it in range(1):
        #print('1')
        try:
            glr = lime_local_results(x_toexplain,
                                     dataset,
                                     dataset_accuracy=X_t,
                                     pred_fn=clf.predict_proba,
                                     label_toexplain=1, num_features=NUM_FEATURES) 
            #print('2')
            klr = lime_local_results(x_toexplain,
                                     dataset,
                                     dataset_accuracy=X_t,
                                     pred_fn=clf.predict_proba,
                                     label_toexplain=1, kernel_width=KW, num_features=NUM_FEATURES)
            #klr = 0
            #print('3')
            prop = prop_local_results(x_toexplain,
                                     dataset,
                                     dataset_accuracy=X_t,
                                     pred_fn=clf.predict_proba,
                                     label_toexplain=1,
                                     radius_train=0.4, num_features=NUM_FEATURES) #test utiliser meme radius pour accuracy et train (centres differents) = 0.3 distance max
            #print('4')

            global_lime_i.append(glr)
            kernel_lime_i.append(klr)
            prop_i.append(prop)
        except ValueError:
            missing += 1
    global_lime_i = np.array(global_lime_i)
    kernel_lime_i = np.array(kernel_lime_i)
    prop_i = np.array(prop_i)
    
    output[d_name] = (global_lime_i, kernel_lime_i, prop_i)
    print(output[d_name])
    print('missing', missing)

print(output)
import pickle
file_out = open('dict_results2.obj', 'wb')
pickle.dump(output, file_out)
