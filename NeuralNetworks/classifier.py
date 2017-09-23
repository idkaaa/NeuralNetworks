print(__doc__)

# Remember to update the script for the new data when you change this URL
#URL = "http://mlr.cs.umass.edu/ml/machine-learning-databases/spambase/spambase.data"
URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

from pandas import read_table
import numpy as np
import matplotlib.pyplot as plt

try:
    # [OPTIONAL] Seaborn makes plots nicer
    import seaborn
except ImportError:
    pass

import numpy as np
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import time




def IrisFeaturesToNumbers(frame):
    frame.replace(
        to_replace = {4: #<-The column to replace 
         {
             'Iris-setosa': '1', 
             'Iris-versicolor': '2',
             'Iris-virginica': '3'
             }},
        inplace=True
        )
    return frame

def p_Plot_Subplots_Learning_Curve(title, train_scores, test_scores, cost_values):
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
   
    plt.title(title)

    ax1.xlabel("Iterations")
    ax1.ylabel("Score")
    
    #number of steps on the X-axis
    train_sizes = np.linspace(1, len(train_scores), len(train_scores))
   
    ax1.grid()
    
    ax1.plot(train_sizes, train_scores, 'o-', color="r",
             label="Training score")
    ax1.plot(train_sizes, test_scores, 'o-', color="g",
             label="Cross-validation score")

    ax1.legend(loc="best")


    ax1.xlabel("Iterations")
    ax1.ylabel("Cost")

    ax2.grid()
    
    ax2.plot(train_sizes, cost_values, 'o-', color="b",
             label="Cost Value")

    ax2.legend(loc="best")


    return plt

def plot_learning_curve(title, train_scores, test_scores):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """

   

    plt.figure()
    plt.title(title)
    plt.xlabel("Iterations")
    plt.ylabel("Score")
    
    #number of steps on the X-axis
    train_sizes = np.linspace(1, len(train_scores), len(train_scores))
   
    plt.grid()
    
    plt.plot(train_sizes, train_scores, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores, 'o-', color="g",
             label="Testing score")

    plt.legend(loc="best")

    return plt

def download_data():
    '''
    Downloads the data for this script into a pandas DataFrame.
    '''

    # If your data is in an Excel file, install 'xlrd' and use
    # pandas.read_excel instead of read_table
    #from pandas import read_excel
    #frame = read_excel(URL)

    # If your data is in a private Azure blob, install 'azure-storage' and use
    # BlockBlobService.get_blob_to_path() with read_table() or read_excel()
    #from azure.storage.blob import BlockBlobService
    #service = BlockBlobService(ACCOUNT_NAME, ACCOUNT_KEY)
    #service.get_blob_to_path(container_name, blob_name, 'my_data.csv')
    #frame = read_table('my_data.csv', ...

    frame = read_table(
        URL,
        
        # Uncomment if the file needs to be decompressed
        #compression='gzip',
        #compression='bz2',

        # Specify the file encoding
        # Latin-1 is common for data from US sources
        encoding='latin-1',
        #encoding='utf-8',  # UTF-8 is also common

        # Specify the separator in the data
        sep=',',            # comma separated values
        #sep='\t',          # tab separated values
        #sep=' ',           # space separated values

        # Ignore spaces after the separator
        skipinitialspace=True,

        # Generate row labels from each row number
        index_col=None,
        #index_col=0,       # use the first column as row labels
        #index_col=-1,      # use the last column as row labels

        # Generate column headers row from each column number
        header=None,
        #header=0,          # use the first line as headers

        # Use manual headers and skip the first row in the file
        #header=0,
        #names=['col1', 'col2', ...],
    )

    # Return a subset of the columns
    #return frame[['col1', 'col4', ...]]

    # Return the entire frame
    return frame

def p_FormatData(DataFrame):
    data = frame.values[:,0:4]
    target = frame.values[:,4]

     # Normalize the attribute values to mean=0 and variance=1
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    scaler.fit(data)
    data = scaler.transform(data)

    return data, target

def p_SplitData(TrainData, TargetData):
    from sklearn.model_selection import train_test_split
    return train_test_split(data, target, test_size=0.33)

def p_GetNeuralNetworkLearningCurve(Classifier, X_train, y_train, X_test, y_test):
    train_scores = []
    test_scores = []
    cost_values = []
    clf = Classifier
    StartTime = time.time()
    for i in range(0, 20):
        clf.fit(X_train, y_train)
        train_score = clf.score(X_train, y_train)
        train_scores.append(train_score)
        test_score = clf.score(X_test, y_test)
        test_scores.append(test_score)
        cost_value = clf.loss_
        cost_values.append(cost_value)
    EndTime = time.time()
    TotalTime = EndTime - StartTime

    return train_scores, test_scores, cost_values, TotalTime

frame = download_data()

data, target = p_FormatData(frame)

X_train, X_test, y_train, y_test = p_SplitData(data, target)

clf = MLPClassifier(
    solver = 'lbfgs',
    alpha = 1e-5,
    hidden_layer_sizes = (3),
    random_state = 1,
    warm_start = True,
    max_iter = 1,
    verbose = True
    )
train_scores, test_scores, cost_values, TotalTime = p_GetNeuralNetworkLearningCurve(clf, X_train, y_train, X_test, y_test)

plot_learning_curve("Neural Network Learning Curve Graph", train_scores, test_scores)

#p_Plot_Subplots_Learning_Curve("Neural Network Learning Curve Graph", train_scores, test_scores, cost_values)

OtherStats = "Total Run Time: {} \n".format(TotalTime)

#Create empty plot with blank marker containing the extra label
plt.plot([], [], ' ', label=OtherStats)


plt.legend()


#TODO: timer

plt.show()
