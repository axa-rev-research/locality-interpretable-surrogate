import pandas, numpy, sklearn
import pylab as plt
import numpy as np
from scipy.spatial.distance import euclidean
from sklearn.metrics import euclidean_distances, accuracy_score
import seaborn as sns

import lime_assessment
import lime_assessment.lime_tabular


from shap import KernelExplainer

from matplotlib.colors import ListedColormap


def get_surrogate_frontier_LIME(x1, explanation, label_toexplain):
    """
    Generate linear regression equation from LIME explanation to plot frontier on the feature space at proba 0.5
    For a 2D feature space (hard coded)...
    """

    # Get feature importance for the classification label_toexplain sorted by feature name
    coefs = [x[1] for x in sorted(explanation.as_list(label_toexplain), key=lambda x:x[0])]

    # Get intercept
    intercept = explanation.intercept[label_toexplain]

    # Compute regression frontier (where prediction proba is 0.5)
    return (0.5 - intercept - x1*coefs[0])/coefs[1]


def plot_classification_contour(X, clf, ax):

    ## Inspired by scikit-learn documentation
    h = .02  # step size in the mesh
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    # Generate mesh
    x_min, x_max = X.iloc[:, 0].min() - .5, X.iloc[:, 0].max() + .5
    y_min, y_max = X.iloc[:, 1].min() - .5, X.iloc[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)

    plt.sca(ax)
    plt.contourf(xx, yy, Z, alpha=.5, cmap=cm)
    


def plot_training_set(X, y, ax):

    X[y==0].plot(x=X.columns[0], y=X.columns[1], kind='scatter', ax=ax, c='lightgray', marker='x', label='Class 0')
    X[y==1].plot(x=X.columns[0], y=X.columns[1], kind='scatter', ax=ax, c='black', marker='x', label='Class 1')
    plt.xlabel('Feature 0')
    plt.ylabel('Feature 1')

def plot_lime_regression(X, explanation, x_toexplain, label_toexplain, ax, color, ptp):

    # Get LIME linear regression
    x_ridge = [x[0] for x in ptp]
    y_ridge = [x[1] for x in ptp]

    # Plot LIME linear regression
    plt.sca(ax)
    plt.plot(x_ridge, y_ridge, color=color, linestyle=':', linewidth=4, label="ridge regression")
    plt.scatter(x_toexplain.iloc[0], x_toexplain.iloc[1], color=color, marker='8', linewidth=4)


def LIME_graph(X, y, feature_names, ylabels, clf, xs_toexplain, labels_toexplain, radius_exp=None, X_global=None, y_global=None, ax=None, verbose=False, subplots=False, kernel_width=None):
    '''
    NB: label_toexplain ne sert à rien pour le moment
    '''

    ## Plot explanations on feature space
    if ax is None:
        if subplots:
            if len(xs_toexplain)>=5:
                nrows = int(len(xs_toexplain)/5)
                fig, axs = plt.subplots(nrows=nrows, ncols=int(len(xs_toexplain)/nrows), figsize=(15,3*nrows))
                axs = axs.flatten()
            else:
                fig, axs = plt.subplots(nrows=1, ncols=len(xs_toexplain), figsize=(15,3))
        else:
            fig, ax = plt.subplots()

    # Plot LIME result - loop if several
    for i in range(len(xs_toexplain)):
        
        if subplots:
            plt.sca(axs[i])
            ax = axs[i]
            
        if i==0 or subplots:
            
            
            if radius_exp is not None:
                # Plot contour of black-box predictions
                plot_classification_contour(X_global, clf, ax)
                # Plot training set
                plot_training_set(X_global, y_global, ax)
                circle = plt.Circle((xs_toexplain[i].iloc[0], xs_toexplain[i].iloc[1]), radius=radius_exp, color='r', fill=False, linewidth=1)
                ax.add_artist(circle)
            else:
                # Plot contour of black-box predictions
                plot_classification_contour(X, clf, ax)
                # Plot training set
                plot_training_set(X, y, ax)
            
            ylim_bak = ax.get_ylim()
            xlim_bak = ax.get_xlim()
            #color_palette = sns.color_palette("bright", n_colors=len(xs_toexplain))
            #color_palette = ['lime' for _ in range(len(xs_toexplain))]
            color_palette = ['blue', 'lime']
        j = 0
        ## LIME - Generate explanations
        if kernel_width == None:
            kernel_width = [None]
        for kw in kernel_width:
            
            explainer = lime_assessment.lime_tabular.LimeTabularExplainer(X, feature_names=feature_names, class_names=ylabels, discretize_continuous=False, kernel_width=kw)
            exp = explainer.explain_instance(xs_toexplain[i], clf.predict_proba, num_features=2, top_labels=len(ylabels), labels=range(len(ylabels)))
            
            # Plot LIME regression
            plot_lime_regression(X, exp, xs_toexplain[i], labels_toexplain[i], ax, color_palette[j], exp.points_to_plot)
            j += 1
        plt.ylim(ylim_bak)
        plt.xlim(xlim_bak)
        #plt.xlim((X.iloc[:,0].min() - 0.5, X.iloc[:,0].max() + 0.5))

    if verbose:
        print(exp.as_list(1))
        print("Predicted class (clf): "+str(clf.predict([xs_toexplain[i].values])))
        print("Predicted probability (LIME reg) to be in the above class (clf)", exp.local_pred)
        print("LIME (genre de) R2: "+str(exp.score))


def get_random_points_within_hypersphere(x_toexplain, r=1, N=100):
    # Generate N random points in a hypersphere of radius r

    res = []

    N_todraw = N

    while len(res) < N:
        N_todraw = N - len(res)

        X_generated = numpy.random.uniform(low=[x_toexplain-r for _ in range(N_todraw)], high=[x_toexplain+r for _ in range(N_todraw)])
        dists = euclidean_distances(x_toexplain.to_frame().T, X_generated)[0]

        for i in range(X_generated.shape[0]):
            # Check if x_generated is within hypersphere (if kind=='hypersphere')
            if dists[i] < r:
                res.append(pandas.Series(X_generated[i], x_toexplain.index))

    X_generated = pandas.DataFrame(res)
    
    return X_generated

def get_LIME_predictions(X_m, clf, exp, label_toexplain=0, regression=False):

    # Normalize X_generated
    X_m_norm = (X_m-exp.mean_)/exp.scale_

    # Get LIME surrogate
    surrogate = exp.easy_model[label_toexplain]

    # Get LIME surrogate prediction
    y_m_surrogate_pred_proba = surrogate.predict(X_m_norm.iloc[:,exp.used_features[label_toexplain]])

    if regression:
        y_m_surrogate_pred = y_m_surrogate_pred_proba
    else:
        # OK for a binary classification problem
        y_m_surrogate_pred = numpy.zeros(X_m_norm.shape[0])
        mask = y_m_surrogate_pred_proba<0.5
        y_m_surrogate_pred[mask] = 1-label_toexplain
        mask = y_m_surrogate_pred_proba>=0.5
        y_m_surrogate_pred[mask] = label_toexplain

    return y_m_surrogate_pred

def get_surrogate_accuracy_hypersphere(x_toexplain, X, clf, ylabels, label_toexplain=0, N=1000, num_features=10):
    
    # Generate radius
    dists = euclidean_distances(x_toexplain.to_frame().T, X)
    dists = pandas.Series(dists[0], index=X.index)
    radius_perc = numpy.arange(2,11)/10.
    radius = radius_perc*dists.max()

    res_accuracy = {}
    
    # Loop over radius
    for i in range(len(radius)):
        r = radius[i]
    
        # Generate points in hypersphere
        X_m = get_random_points_within_hypersphere(x_toexplain, r=r, N=N)
        
        # Get LIME predictions on LIME trained on the entire dataset X
        explainer = lime_assessment.lime_tabular.LimeTabularExplainer(X, discretize_continuous=False)
        exp = explainer.explain_instance(x_toexplain, clf.predict_proba, num_features=num_features, top_labels=len(ylabels), labels=range(len(ylabels)))
    
        y_m_surrogate_pred = get_LIME_predictions(X_m, clf, exp, label_toexplain=label_toexplain, regression=False)
        
        # Get blackbox predictions
        y_m_blackbox = clf.predict(X_m)
        
        # Compute score
        res_accuracy[radius_perc[i]] = accuracy_score(y_m_blackbox, y_m_surrogate_pred)
    
    return pandas.Series(res_accuracy)

def get_surrogate_accuracy_hypersphere_kernelvariation(x_toexplain, X, clf, ylabels, label_toexplain=0, N=1000, num_features=10):
    
    # Generate radius
    dists = euclidean_distances(x_toexplain.to_frame().T, X)
    dists = pandas.Series(dists[0], index=X.index)
    radius_perc = 0.5 #numpy.arange(2,11)/10.
    radius = radius_perc*dists.max()
    kernel_widths = [0.3, 0.5, None, 2, 5, 1000]
    
    res_accuracy = {}
    
    # Loop over radius
    for i in range(len(kernel_widths)):
        kw = kernel_widths[i]
    
        # Generate points in hypersphere
        X_m = get_random_points_within_hypersphere(x_toexplain, r=radius, N=N)
        
        # Get LIME predictions on LIME trained on the entire dataset X
        explainer = lime_assessment.lime_tabular.LimeTabularExplainer(X, discretize_continuous=False, kernel_width=kw)
        exp = explainer.explain_instance(x_toexplain, clf.predict_proba, num_features=num_features, top_labels=len(ylabels), labels=range(len(ylabels)))
    
        y_m_surrogate_pred = get_LIME_predictions(X_m, clf, exp, label_toexplain=label_toexplain, regression=False)
        
        # Get blackbox predictions
        y_m_blackbox = clf.predict(X_m)
        
        # Compute score
        if kw == None:
            kernel_widths[i] = 'Default LIME kernel'
        res_accuracy[kernel_widths[i]] = accuracy_score(y_m_blackbox, y_m_surrogate_pred)
    
    return pandas.Series(res_accuracy)

def get_surrogate_accuracy_growing_hypersphere(xs_toexplain, X, clf, ylabels, label_toexplain=0, N=1000, num_features=10):
    
    res_accuracy = {}
    for i in range(len(xs_toexplain)):
        x_toexplain = xs_toexplain[i]
        res_accuracy[i] = get_surrogate_accuracy_hypersphere(x_toexplain, X, clf, ylabels, label_toexplain=label_toexplain, N=N, num_features=num_features)

    res_accuracy = pandas.DataFrame(res_accuracy)
    res_accuracy.columns = ['Point '+str(i) for i in range(res_accuracy.shape[1])]
    
    return res_accuracy


def get_surrogate_accuracy_growing_kernelwidth(xs_toexplain, X, clf, ylabels, label_toexplain=0, N=1000, num_features=10):
    
    res_accuracy = {}
    for i in range(len(xs_toexplain)):
        x_toexplain = xs_toexplain[i]
        res_accuracy[i] = get_surrogate_accuracy_hypersphere_kernelvariation(x_toexplain, X, clf, ylabels, label_toexplain=label_toexplain, N=N, num_features=num_features)

    res_accuracy = pandas.DataFrame(res_accuracy)
    res_accuracy.columns = ['Point '+str(i) for i in range(res_accuracy.shape[1])]
    
    return res_accuracy



def LIME_graphSHAP(X, y, feature_names, ylabels, clf, xs_toexplain, labels_toexplain, ax=None, subplots=False, plotlime=True):

    ## Plot explanations on feature space
    if ax is None:
        if subplots:
            if len(xs_toexplain)>=5:
                nrows = int(len(xs_toexplain)/5)
                fig, axs = plt.subplots(nrows=nrows, ncols=int(len(xs_toexplain)/nrows), figsize=(15,3*nrows))
                axs = axs.flatten()
            else:
                fig, axs = plt.subplots(nrows=1, ncols=len(xs_toexplain), figsize=(15,3))
        else:
            fig, ax = plt.subplots()

    # Plot LIME result - loop if several
    for i in range(len(xs_toexplain)):
        
        if subplots:
            plt.sca(axs[i])
            ax = axs[i]
            
        if i==0 or subplots:
            

            # Plot contour of black-box predictions
            plot_classification_contour(X, clf, ax)
            # Plot training set
            plot_training_set(X, y, ax)

            ylim_bak = ax.get_ylim()
            xlim_bak = ax.get_xlim()
            #color_palette = sns.color_palette("bright", n_colors=len(xs_toexplain))
            color_palette = ['lime' for _ in range(len(xs_toexplain))]
        
        ## LIME - Generate explanations
        explainer = lime_assessment.lime_tabular.LimeTabularExplainer(X, feature_names=feature_names, class_names=ylabels, discretize_continuous=False, kernel_width=None)
        exp = explainer.explain_instance(xs_toexplain[i], clf.predict_proba, num_features=2, top_labels=len(ylabels), labels=range(len(ylabels)))
        
        shap_explainer = KernelExplainer(clf.predict_proba, X, nsamples=10000)
        e = shap_explainer.explain(np.reshape(xs_toexplain[i], (1, X.shape[1])))

        # Plot LIME regression
        if plotlime == True:
            plot_lime_regression(X, exp, xs_toexplain[i], labels_toexplain[i], ax, color_palette[i], exp.points_to_plot)
        x_ridge = [-10, 10]
        row = 0
        y_shap = [(0.5 - e.effects[0, row] * x - e.base_value[row])/e.effects[1, row] for x in x_ridge]


        # Plot LIME linear regression
        plt.sca(ax)
        plt.plot(x_ridge, y_shap, color='red', linestyle=':', linewidth=4, label="other shap regression")

        plt.scatter(xs_toexplain[i][0], xs_toexplain[i][1], color='lime', marker='8', linewidth=4)
        plt.ylim(ylim_bak)
        plt.xlim(xlim_bak)
        #plt.xlim((X.iloc[:,0].min() - 0.5, X.iloc[:,0].max() + 0.5))
