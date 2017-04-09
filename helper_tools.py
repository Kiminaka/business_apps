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
from sklearn.metrics import precision_recall_curve, average_precision_score
import itertools
from itertools import cycle
from scipy import interp
from matplotlib import use
use("Agg")
import csv,argparse

def run_classifier_metrics(X, y,classifier,k):
    accuracy = cross_val_score(classifier, X, y, scoring='accuracy', cv=k).mean()
    precision = cross_val_score(classifier, X, y, scoring='precision', cv=k ).mean()
    recall = cross_val_score(classifier, X, y, scoring='recall', cv=k).mean()
    f1 = cross_val_score(classifier, X, y, scoring='f1', cv=k).mean()
    roc_auc = cross_val_score(classifier, X, y, scoring='roc_auc', cv=k).mean()
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
    run_classifier_metrics(X, y,classifier,k)

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

def get_roc(df,score,target,title,plot=1):
    df1 = df[[score,target]].dropna()
    fpr, tpr, thresholds = roc_curve(df1[target], df1[score])
    ks=np.abs(tpr-fpr)
    if plot==1:
    # Plot ROC curve
        plt.figure(figsize=(6,4))
        plt.plot(fpr, tpr, label='AUC=%0.2f KS=%0.2f' %(auc(fpr, tpr),ks.max()))
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.grid(b=True, which='both', color='0.65',linestyle='-')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title+'Receiver Operating Characteristic')
        plt.legend(loc="lower right")
    return auc(fpr, tpr),np.max(np.abs(tpr-fpr)),thresholds[ks.argmax()]

def get_cum_gains(df,score,target,title):
    df1 = df[[score,target]].dropna()
    fpr, tpr, thresholds = roc_curve(df1[target], df1[score])
    ppr=(tpr*df[target].sum()+fpr*(df[target].count()-df[target].sum()))/df[target].count()
    plt.figure(figsize=(12,4))
    plt.subplot(1,2,1)
    plt.plot(ppr, tpr, label='')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
    plt.xlabel('%Population')
    plt.ylabel('%Target')
    plt.title(title+'Cumulative Gains Chart')
    plt.legend(loc="lower right")

    plt.subplot(1,2,2)
    plt.plot(ppr, tpr/ppr, label='')
    plt.plot([0, 1], [1, 1], 'k--')
    plt.grid(b=True, which='both', color='0.65',linestyle='-')
    plt.xlabel('%Population')
    plt.ylabel('Lift')
    plt.title(title+'Lift Curve')

def get_precision_recall(df,score,target,title):
    precision, recall, _ = precision_recall_curve(df[target], df[score])
    roc_pr = average_precision_score(df[target], df[score])
    # Plot ROC curve
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label='Precision-Recall curve (AUC = %0.2f)' % roc_pr)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(title+"Precision-Recall Curve")
    plt.legend(loc="lower left")
    plt.grid(b=True, which='both', color='0.65',linestyle='-')

def get_deciles_analysis(df,score,target):
    df1 = df[[score,target]].dropna()
    _,bins = pd.qcut(df1[score],10,retbins=True)
    bins[0] -= 0.001
    bins[-1] += 0.001
    bins_labels = ['%d.(%0.2f,%0.2f]'%(9-x[0],x[1][0],x[1][1]) for x in enumerate(zip(bins[:-1],bins[1:]))]
    bins_labels[0] = bins_labels[0].replace('(','[')
    df1['Decile']=pd.cut(df1[score],bins=bins,labels=bins_labels)
    df1['Population']=1
    df1['Zeros']=1-df1[target]
    df1['Ones']=df1[target]
    summary=df1.groupby(['Decile'])[['Ones','Zeros','Population']].sum()
    summary=summary.sort_index(ascending=False)
    summary['TargetRate']=summary['Ones']/summary['Population']
    summary['CumulativeTargetRate']=summary['Ones'].cumsum()/summary['Population'].cumsum()
    summary['TargetsCaptured']=summary['Ones'].cumsum()/summary['Ones'].sum()
    return summary

def confusion_matrix(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    num_ratings = int(max_rating - min_rating + 1)
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a, b in zip(rater_a, rater_b):
        conf_mat[a - min_rating][b - min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None:
        min_rating = min(ratings)
    if max_rating is None:
        max_rating = max(ratings)
    num_ratings = int(max_rating - min_rating + 1)
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
        hist_ratings[r - min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the quadratic weighted kappa
    quadratic_weighted_kappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    quadratic_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    rater_a = np.array(rater_a, dtype=int)
    rater_b = np.array(rater_b, dtype=int)
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(min(rater_a), min(rater_b))
    if max_rating is None:
        max_rating = max(max(rater_a), max(rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator

def linear_weighted_kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the linear weighted kappa
    linear_weighted_kappa calculates the linear weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    linear_weighted_kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    linear_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = abs(i - j) / float(num_ratings - 1)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator

def kappa(rater_a, rater_b, min_rating=None, max_rating=None):
    """
    Calculates the kappa
    kappa calculates the kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    kappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
    kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(rater_a + rater_b)
    if max_rating is None:
        max_rating = max(rater_a + rater_b)
    conf_mat = confusion_matrix(rater_a, rater_b,
                                min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            if i == j:
                d = 0.0
            else:
                d = 1.0
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items

    return 1.0 - numerator / denominator

def mean_quadratic_weighted_kappa(kappas, weights=None):
    """
    Calculates the mean of the quadratic
    weighted kappas after applying Fisher's r-to-z transform, which is
    approximately a variance-stabilizing transformation.  This
    transformation is undefined if one of the kappas is 1.0, so all kappa
    values are capped in the range (-0.999, 0.999).  The reverse
    transformation is then applied before returning the result.
    mean_quadratic_weighted_kappa(kappas), where kappas is a vector of
    kappa values
    mean_quadratic_weighted_kappa(kappas, weights), where weights is a vector
    of weights that is the same size as kappas.  Weights are applied in the
    z-space
    """
    kappas = np.array(kappas, dtype=float)
    if weights is None:
        weights = np.ones(np.shape(kappas))
    else:
        weights = weights / np.mean(weights)

    # ensure that kappas are in the range [-.999, .999]
    kappas = np.array([min(x, .999) for x in kappas])
    kappas = np.array([max(x, -.999) for x in kappas])

    z = 0.5 * np.log((1 + kappas) / (1 - kappas)) * weights
    z = np.mean(z)
    return (np.exp(2 * z) - 1) / (np.exp(2 * z) + 1)

def weighted_mean_quadratic_weighted_kappa(solution, submission):
    predicted_score = submission[submission.columns[-1]].copy()
    predicted_score.name = "predicted_score"
    if predicted_score.index[0] == 0:
        predicted_score = predicted_score[:len(solution)]
        predicted_score.index = solution.index
    combined = solution.join(predicted_score, how="left")
    groups = combined.groupby(by="essay_set")
    kappas = [quadratic_weighted_kappa(group[1]["essay_score"], group[1]["predicted_score"]) for group in groups]
    weights = [group[1]["essay_weight"].irow(0) for group in groups]
    return mean_quadratic_weighted_kappa(kappas, weights=weights)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ifile", help="Input file")
    parser.add_argument("--d", help="Delimiter. Default: Comma")
    parser.add_argument("--score", help="Score Column. Default: score")
    parser.add_argument("--target", help="Target Column. Default: target")
    parser.add_argument("--tag", help="Output Files Tag. Default: performance")
    parser.add_argument("--title", help="Charts Title. Default: None")
    
    args = parser.parse_args()
    infile = args.ifile
    score = args.score if args.score else 'score'
    target = args.target if args.target else 'target'
    tag = args.tag if args.tag else 'performance'
    delimiter = args.d if args.d else ','
    title = args.title+':' if args.title else ''
    
    score_card=pd.read_csv(infile,delimiter=delimiter,usecols=[score,target])
    auc,ks,ks_score=get_roc(score_card,score,target,title)
    plt.savefig('%s_roc.png'%tag)
    get_cum_gains(score_card,score,target,title)
    plt.savefig('%s_cum_gains.png'%tag)
    get_precision_recall(score_card,score,target,title)
    plt.savefig('%s_precision_recall.png'%tag)
    decile_analysis=get_deciles_analysis(score_card,score,target)
    decile_analysis.to_csv('%s_decile_analysis.csv'%tag)
    pd.Series([auc,ks,ks_score],index=['auc','ks','ks_score']).to_csv('%s_summary.csv'%tag)
    
