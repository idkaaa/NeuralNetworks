print(__doc__)

# Author: Eustache Diemert <eustache@diemert.fr>
# License: BSD 3 clause

import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.parasite_axes import host_subplot
from mpl_toolkits.axisartist.axislines import Axes
from scipy.sparse.csr import csr_matrix

from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.svm.classes import NuSVR
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor
from sklearn.linear_model.stochastic_gradient import SGDClassifier
from sklearn.metrics import hamming_loss

from sklearn.neural_network import MLPClassifier

# #############################################################################
# Routines


# Initialize random generator
np.random.seed(0)

def p_IrisFeaturesToNumbers(frame):
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

def p_FormatData(frame):

    frame = p_IrisFeaturesToNumbers(frame)

    data = frame.values[:,0:4]
    target = frame.values[:,4]

     # Normalize the attribute values to mean=0 and variance=1
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()

    scaler.fit(data)
    data = scaler.transform(data)

    return data, target

def p_SplitData(Data, Target):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(Data, Target, test_size=0.33)
    data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train,
            'y_test': y_test}
    return data


def p_GetData():
    from pandas import read_table
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
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


def generate_data(case, sparse=False):
    """Generate regression/classification data."""
    bunch = None
    if case == 'regression':
        bunch = datasets.load_boston()
    elif case == 'classification':
        bunch = datasets.fetch_20newsgroups_vectorized(subset='all', data_home='C:\\School\\ML\\20News\\')
    X, y = shuffle(bunch.data, bunch.target)
    offset = int(X.shape[0] * 0.8)
    X_train, y_train = X[:offset], y[:offset]
    X_test, y_test = X[offset:], y[offset:]
    if sparse:
        X_train = csr_matrix(X_train)
        X_test = csr_matrix(X_test)
    else:
        X_train = np.array(X_train)
        X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_train = np.array(y_train)
    data = {'X_train': X_train, 'X_test': X_test, 'y_train': y_train,
            'y_test': y_test}
    return data


def benchmark_influence(conf):
    """
    Benchmark influence of :changing_param: on both MSE and latency.
    """
    prediction_times = []
    prediction_powers = []
    complexities = []
    for param_value in conf['changing_param_values']:
        conf['tuned_params'][conf['changing_param']] = param_value
        estimator = conf['estimator'](**conf['tuned_params'])
        print("Benchmarking %s" % estimator)
        estimator.fit(conf['data']['X_train'], conf['data']['y_train'])
        conf['postfit_hook'](estimator)
        complexity = conf['complexity_computer'](estimator)
        complexities.append(complexity)
        start_time = time.time()
        for _ in range(conf['n_samples']):
            y_pred = estimator.predict(conf['data']['X_test'])
        elapsed_time = (time.time() - start_time) / float(conf['n_samples'])
        prediction_times.append(elapsed_time)
        pred_score = conf['prediction_performance_computer'](
            conf['data']['y_test'], y_pred)
        prediction_powers.append(pred_score)
        print("Complexity: %d | %s: %.4f | Pred. Time: %fs\n" % (
            complexity, conf['prediction_performance_label'], pred_score,
            elapsed_time))
    return prediction_powers, prediction_times, complexities


def plot_influence(conf, mse_values, prediction_times, complexities):
    """
    Plot influence of model complexity on both accuracy and latency.
    """
    plt.figure(figsize=(12, 6))
    host = host_subplot(111, axes_class=Axes)
    plt.subplots_adjust(right=0.75)
    par1 = host.twinx()
    host.set_xlabel('Model Complexity (%s)' % conf['complexity_label'])
    y1_label = conf['prediction_performance_label']
    y2_label = "Time (s)"
    host.set_ylabel(y1_label)
    par1.set_ylabel(y2_label)
    p1, = host.plot(complexities, mse_values, 'b-', label="prediction error")
    p2, = par1.plot(complexities, prediction_times, 'r-',
                    label="latency")
    host.legend(loc='upper right')
    host.axis["left"].label.set_color(p1.get_color())
    par1.axis["right"].label.set_color(p2.get_color())
    plt.title('Influence of Model Complexity - %s' % conf['estimator'].__name__)
    plt.show()


def _count_nonzero_coefficients(estimator):
    a = estimator.coef_.toarray()
    return np.count_nonzero(a)

# #############################################################################
# Main code
#regression_data = generate_data('regression')
#classification_data = generate_data('classification', sparse=True)
#configurations = [
#    {'estimator': SGDClassifier,
#     'tuned_params': {'penalty': 'elasticnet', 'alpha': 0.001, 'loss':
#                      'modified_huber', 'fit_intercept': True},
#     'changing_param': 'l1_ratio',
#     'changing_param_values': [0.25, 0.5, 0.75, 0.9],
#     'complexity_label': 'non_zero coefficients',
#     'complexity_computer': _count_nonzero_coefficients,
#     'prediction_performance_computer': hamming_loss,
#     'prediction_performance_label': 'Hamming Loss (Misclassification Ratio)',
#     'postfit_hook': lambda x: x.sparsify(),
#     'data': classification_data,
#     'n_samples': 30},
#    {'estimator': NuSVR,
#     'tuned_params': {'C': 1e3, 'gamma': 2 ** -15},
#     'changing_param': 'nu',
#     'changing_param_values': [0.1, 0.25, 0.5, 0.75, 0.9],
#     'complexity_label': 'n_support_vectors',
#     'complexity_computer': lambda x: len(x.support_vectors_),
#     'data': regression_data,
#     'postfit_hook': lambda x: x,
#     'prediction_performance_computer': mean_squared_error,
#     'prediction_performance_label': 'MSE',
#     'n_samples': 30},
#    {'estimator': GradientBoostingRegressor,
#     'tuned_params': {'loss': 'ls'},
#     'changing_param': 'n_estimators',
#     'changing_param_values': [10, 50, 100, 200, 500],
#     'complexity_label': 'n_trees',
#     'complexity_computer': lambda x: x.n_estimators,
#     'data': regression_data,
#     'postfit_hook': lambda x: x,
#     'prediction_performance_computer': mean_squared_error,
#     'prediction_performance_label': 'MSE',
#     'n_samples': 30},
#]

PandasDataframe = p_GetData()
Data, Classes = p_FormatData(PandasDataframe)
classification_data = p_SplitData(Data, Classes)

configurations = [
    {
        'estimator': MLPClassifier,
        'tuned_params': 
        {
            'solver': 'lbfgs', 
            'alpha': 1e-5,
            'max_iter': 20,
        },
        'changing_param': 'hidden_layer_sizes',
        'changing_param_values': [3, 6, 9, 12, 15],
        'complexity_label': 'Number of Hidden Units in Single Hidden Layer',
        'complexity_computer': lambda x: x.hidden_layer_sizes,
        'data': classification_data,
        'postfit_hook': lambda x: x,
        'prediction_performance_computer': mean_squared_error,
        'prediction_performance_label': 'MSE',
        'n_samples':30
    },
]
for conf in configurations:
    prediction_performances, prediction_times, complexities = \
        benchmark_influence(conf)
    plot_influence(conf, prediction_performances, prediction_times,
                   complexities)
