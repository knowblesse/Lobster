import sklearn
if (sklearn.__version__ != '0.23.2'):
    raise Exception("scikit-learn package version must be 0.23.2")
import os
import  numpy as np
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

for datanum in np.arange(8):

    # Load .mat data
    BASE_PATH = r'C:\Users\knowb\SynologyDrive\20JUN'
    datalist = os.listdir(BASE_PATH)
    data = loadmat(os.path.join(BASE_PATH, datalist[datanum]))
    print(datalist[datanum] + ' is loaded \n')
    X = data.get('X')
    Y = data.get('y')
    Y = np.squeeze(Y)
    #np.random.shuffle(Y)

    Y_label = ['Head Entry', 'Avoidance', 'Escape']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.5, stratify=Y)

    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.naive_bayes import CategoricalNB
    param_grid = {'C' : np.linspace(1,3,21), 'kernel': ['poly','rbf'], 'gamma' : ['auto','scale'], 'tol' : [1e-3, 1e-4, 1e-5]}
    #param_grid = {'alpha' : np.linspace(0, 1, 11)}

    scores = ['accuracy','precision', 'recall']
    ## Parameter search

    print('Hyper parameter tuning for accuracy')
    print()
    search = GridSearchCV(SVC(), iid=False,  param_grid=param_grid, cv=3, n_jobs=-1, scoring='balanced_accuracy')
    #search = GridSearchCV(CategoricalNB(), param_grid=param_grid, cv=5, n_jobs=-1, scoring='balanced_accuracy')
    search.fit(X_train, Y_train)

    print("Grid scores on development set:")
    print()
    means = search.cv_results_['mean_test_score']
    stds = search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    Y_true, Y_pred = Y_test, search.predict(X_test)
    print(classification_report(Y_true, Y_pred))
    print()
    print('Best parameter')
    print(search.best_params_)
    print('Ratio')
    print("%d %d %d" %(np.sum(Y_test == 1), np.sum(Y_test ==2), np.sum(Y_test ==3)))

    # Classification Result
    confusion_mat = confusion_matrix(Y_true, Y_pred,normalize='true') # row is actual. # column is predicted
    cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
    f, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(confusion_mat, cmap=cmap, vmin=0, vmax=1, annot=True, square=True, linewidths=.5, cbar_kws={"shrink": .5}, xticklabels=Y_label, yticklabels=Y_label)
    ax.set_xlabel('predicted')
    ax.set_ylabel('actual')
    plt.show()


