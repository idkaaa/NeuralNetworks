print(__doc__)

# Uncomment this call when using matplotlib to generate images
# rather than displaying interactive UI.
#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import time

#import BanknotesData as MyData
#import PimaIndiansData as MyData

#
# This method plots the actual curve.
# 
def p_PlotComplexityCurve(title, train_scores, test_scores, LossList):

    figure = plt.figure()
    plt.title(title)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Training and Testing Scores")
    
    #number of steps on the X-axis
    train_sizes = np.linspace(1, len(train_scores), len(train_scores))
   
    plt.grid()
    
    plt.plot(train_sizes, train_scores, '-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores, '-', color="g",
             label="Testing score")

    plt.legend(loc="best")
    figure.tight_layout()
    return plt

#
# This method classifies the data and gathers the data for model complexity
# Analysis. It expects a classifier with a maximum iteration of one and steps
# through each iteration and captures the score data.
#
def p_GetModelComplexityCurveData(Classifier, X_train, y_train, X_test, y_test):
    train_scores = []
    p_ElapsedTrainingTime = 0.0
    test_scores = []
    p_ElapsedTestingTime = 0.0
    test_times = []
    LossList = []
    clf = Classifier

    # For each iteration through the classifier...
    for i in range(0, 300):
        
        StartTrainTime = time.time()
        clf.fit(X_train, y_train)
        p_ElapsedTrainingTime += time.time() - StartTrainTime
        
        StartTestTime = time.time()
        train_score = clf.score(X_train, y_train)
        train_scores.append(train_score)
        test_score = clf.score(X_test, y_test)
        test_scores.append(test_score)
        p_ElapsedTestingTime += time.time() - StartTestTime

        cost_value = clf.loss_
        LossList.append(cost_value)
        print("Training Score: %.3f | Test Score: %.3f | Loss: %f\n" % (
            train_score, test_score, cost_value))

    return train_scores, test_scores, LossList, p_ElapsedTrainingTime, p_ElapsedTestingTime

###############################################################################
#
# This is the main method area.
#
###############################################################################

# Get the data ready
frame = MyData.download_data()
data, target = MyData.p_FormatData(frame)
X_train, X_test, y_train, y_test = MyData.p_SplitData(data, target)

# Initialize the configuration of the classifier
#clf = MLPClassifier(
#    solver = 'sgd',
#    hidden_layer_sizes = (50),
#    random_state = 1,
#    max_iter = 1,
#    warm_start = True,
#    )
clf = MLPClassifier(
    solver = 'sgd',
    hidden_layer_sizes = (50),
    random_state = 1,
    max_iter = 1,
    warm_start = True,
    )

# Fit and get results from the data
train_scores, test_scores, LossList, TotalTrainTime, TotalTestTime = p_GetModelComplexityCurveData(clf, X_train, y_train, X_test, y_test)

# Format the parameters to be displayed on the title of the graph 
import json
from textwrap import wrap
ParameterString = json.dumps(clf.get_params())
ParameterString = "\n".join(wrap(ParameterString))

# Plot the graph
p_PlotComplexityCurve("Neural Network Learning Curve Graph: \n" + ParameterString, train_scores, test_scores, LossList)

#Create empty plot with blank marker containing the extra label in the legend
OtherStats = "Total Training Time: %.2fs\nTotal TestingTime: %.2fs" % (TotalTrainTime, TotalTestTime)
plt.plot([], [], ' ', label=OtherStats)
plt.legend()

# finally, show the plot
plt.show()
