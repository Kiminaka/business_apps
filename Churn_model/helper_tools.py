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



def forest_importance_plot(data_frame, train_x, train_y, clf_class=RandomForestClassifier()):
    """
    returns feature importance ranking plot.
    data_frame = pandas dataframe
    clf_class = algorithms that have feature_importances_ are usable. Example: RandomForestClassifier()

    """
    feat_labels = data_frame.columns[:-1]

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


def run_performance(model, X_full, y_full, train_x, train_y, test_x, test_y):
    model = model.fit(train_x, train_y)
    accuracy = cross_val_score(model, X_full, y_full, scoring='accuracy', cv=10).mean()
    precision = cross_val_score(model, X_full, y_full, scoring='precision', cv=10 ).mean()
    recall = cross_val_score(model, X_full, y_full, scoring='recall', cv=10).mean()
    f1 = cross_val_score(model, X_full, y_full, scoring='f1', cv=10).mean()
    roc_auc = cross_val_score(model, X_full, y_full, scoring='roc_auc', cv=10).mean()
    print("-----------------Cross validation scores----------------- \n accuracy {0} \n precision {1} \n recall {2} \n f1 {3} \n roc score {3} \n".format(accuracy, precision, recall, f1, roc_auc))

    print "-----------------Non cross validation scores using test data-----------------"

    print 'Confution matrix \n', confusion_matrix(test_y, model.predict(test_x))
    print
    target_names = ['not churn', 'churn']
    print(classification_report(test_y, model.predict(test_x), target_names=target_names))
    try:
        pred_probas = model.predict_proba(test_x)[:,1]

        fpr,tpr,_ = roc_curve(test_y, pred_probas)
        roc_auc = auc(fpr,tpr)
        plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.legend(loc='lower right')
        plt.show()
    except AttributeError:
        print "ROC curve: predict_proba is not available for the model \n"


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
