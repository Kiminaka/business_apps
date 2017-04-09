import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import cross_val_score
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.learning_curve import validation_curve
from sklearn.model_selection import StratifiedKFold
import itertools
from itertools import cycle
from scipy import interp

def run_classifier_metrics(X, y,classifier):
    accuracy = cross_val_score(classifier, X, y, scoring='accuracy', cv=10).mean()
    precision = cross_val_score(classifier, X, y, scoring='precision', cv=10 ).mean()
    recall = cross_val_score(classifier, X, y, scoring='recall', cv=10).mean()
    f1 = cross_val_score(classifier, X, y, scoring='f1', cv=10).mean()
    roc_auc = cross_val_score(classifier, X, y, scoring='roc_auc', cv=10).mean()
    print("-----------------Cross validation scores----------------- \n accuracy {0} \n precision {1} \n recall {2} \n f1 {3} \n roc score {3} \n".format(accuracy, precision, recall, f1, roc_auc))

def run_cross_val_confusion(X, y,classifier,k=5):

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    cv = StratifiedKFold(n_splits=k)
    colors = cycle(['cyan', 'indigo', 'seagreen', 'yellow', 'blue', 'darkorange'])
    lw = 2
    i = 0

    y_true = []
    y_pred = []
    plt.figure(figsize=(5,5))
    for (train, test), color in zip(cv.split(X, y), colors):
        y_true.append(list(y.ix[test]))
        y_pred.append(classifier.fit(X.ix[train], y.ix[train]).predict(X.ix[test]))

        probas_ = classifier.fit(X.ix[train], y.ix[train]).predict_proba(X.ix[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y.ix[test], probas_[:, 1])
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=lw, color=color,
                 label='ROC fold %d (area = %0.2f)' % (i, roc_auc))

        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=lw, color='k',
             label='Random')
    y_true_unnest = list(itertools.chain(*y_true))
    y_pred_unnest = list(itertools.chain(*y_pred))

    mean_tpr /= cv.get_n_splits(X, y)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--',
             label='Mean ROC (area = %0.2f)' % mean_auc, lw=lw)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0)
    plt.show()
    return y_true_unnest,y_pred_unnest

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 1.3
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def run_performance(X, y,classifier, classes,k):
    run_classifier_metrics(X, y,classifier)

    y_true_unnest, y_pred_unnest = run_cross_val_confusion(X, y,classifier,k)

    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_true_unnest, y_pred_unnest)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=classes,
                          title='Confusion matrix, without normalization')
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix, classes=classes, normalize=True,
                          title='Normalized confusion matrix')

def forest_importance_plot(data_frame, train_x, train_y, clf_class=RandomForestClassifier()):
    """
    returns feature importance ranking plot.
    data_frame = pandas dataframe
    clf_class = algorithms that have feature_importances_ are usable. Example: RandomForestClassifier()

    """
    feat_labels = data_frame.columns[1:]

    forest = clf_class

    forest.fit(train_x, train_y)
    importances = forest.feature_importances_

    indices = np.argsort(importances)[::-1]

    for f in range(train_x.shape[1]):
        print("%2d) %-*s %f" % (f + 1, 30,
                                feat_labels[indices[f]],
                                importances[indices[f]]))

    plt.figure(figsize=(20,10))
    plt.title('Feature Importances')
    plt.bar(range(train_x.shape[1]),
            importances[indices],
            color='lightblue',
            align='center')
    plt.xticks(range(train_x.shape[1]),
               feat_labels[indices], rotation=90)
    plt.xlim([-1, train_x.shape[1]])
    plt.tight_layout()
    plt.show()

def optimal_number_of_features_plot(clf_class, train_x, train_y):
    """
    returns a plot to find out optimal number of features
    clf = any algorithms can be used. Example: RandomForestClassifier()
    """
    clf = RandomForestClassifier()
    rfecv = RFECV(estimator=clf, step=1, cv=5,scoring='accuracy')
    rfecv.fit(train_x, train_y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    plt.figure(figsize=(20,10))
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def learning_curve_plot(clf_class, x, y, standardization=False):
    """
    returns learning curve plot
    clf = any algorithms can be used. Example: RandomForestClassifier()
    standardization = True or False. False as default. Better to standardize features before trainning for many algorithms
    """
    clf = clf_class
    if standardization:
        clf = Pipeline([('scl', StandardScaler()),
                    ('clf', clf_class)])
    train_sizes, train_scores, test_scores =\
                    learning_curve(estimator=clf,
                    X=x,
                    y=y,
                    train_sizes=np.linspace(0.1, 1.0, 10),
                    cv=10,
                    n_jobs=-1)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(train_sizes, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(train_sizes,
                     train_mean + train_std,
                     train_mean - train_std,
                     alpha=0.15, color='blue')

    plt.plot(train_sizes, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(train_sizes,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xlabel('Number of training samples')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.show()


def validation_curve_plot(clf_class, x, y, parameter_range, parameter_name):
    """
    returns validation curve plot
    clf = any algorithms can be used. Example: RandomForestClassifier()
    standardization = True or False. False as default. Better to standardize features before trainning for many algorithms
    """
    clf = clf_class
    param_range = parameter_range
    train_scores, test_scores = validation_curve(
                    estimator=clf,
                    X=x,
                    y=y,
                    param_name=parameter_name,
                    param_range=param_range,
                    cv=10)

    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.plot(param_range, train_mean,
             color='blue', marker='o',
             markersize=5, label='training accuracy')

    plt.fill_between(param_range, train_mean + train_std,
                     train_mean - train_std, alpha=0.15,
                     color='blue')

    plt.plot(param_range, test_mean,
             color='green', linestyle='--',
             marker='s', markersize=5,
             label='validation accuracy')

    plt.fill_between(param_range,
                     test_mean + test_std,
                     test_mean - test_std,
                     alpha=0.15, color='green')

    plt.grid()
    plt.xscale('log')
    plt.legend(loc='lower right')
    plt.xlabel('Parameter C')
    plt.ylabel('Accuracy')
    plt.ylim([0.8, 1.0])
    plt.tight_layout()
    plt.show()
