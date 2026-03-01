import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, GridSearchCV
import data 
import models 


# This is the main function
def main():

    # create the predictors 
    x1 = data.create_data()[0]
    x2 = data.create_data()[1]
    y = data.create_data()[2]

    # Transform into a pandas df and separate predictors from response
    data_df = pd.DataFrame(data={"x1":x1, "x2": x2, "y": y})
    X = data_df[["x1", "x2"]]
    Y = data_df["y"]

    # Plot the observations colored according to their class labels 
    plt.scatter(x1[y], x2[y], label="y = True")
    plt.scatter(x1[~y], x2[~y], label="y = False")

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("x1 and x2")
    plt.legend()
    #plt.show()

    # Fit the logistic regression 
    clf = models.logistic_regression().fit(X, Y)
    Y_pred = clf.predict(X)

    # Confusion matrix of the results 
    m = confusion_matrix(Y, Y_pred)

    # Percentage of correct predictions 
    correct_rate = np.mean(Y_pred == Y)

    # Plot the predictions and color them according to the class label
    correct = (Y_pred == Y.to_numpy())
    
    plt.figure()
    plt.scatter(data_df.loc[correct, "x1"], data_df.loc[correct, "x2"], c="green", label="correct")
    plt.scatter(data_df.loc[~correct, "x1"], data_df.loc[~correct, "x2"], c="red", label="wrong")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Predictions: correct (green) vs wrong (red)")
    plt.legend()
    #plt.show()

    # Other functions of the predictors as predictors: 
    # x1^2 and x2^2
    x1_2 = x1**2 
    x2_2 = x2**2

    # create a new dataframe 
    data_df = pd.DataFrame(data={"x1":x1, "x2": x2, "x1^2": x1_2, "x2^2": x2_2, "y": y})
    X = data_df[["x1", "x2", "x1^2", "x2^2"]]

    # Fit the logistic regression 
    clf = models.logistic_regression().fit(X, Y)
    Y_pred = clf.predict(X)

    # Confusion matrix of the results 
    m = confusion_matrix(Y, Y_pred)

    # Percentage of correct predictions 
    correct_rate = np.mean(Y_pred == Y)

    # Plot the predictions and color them according to the class label
    correct = (Y_pred == Y.to_numpy())
    
    plt.figure()
    plt.scatter(data_df.loc[correct, "x1"], data_df.loc[correct, "x2"], c="green", label="correct")
    plt.scatter(data_df.loc[~correct, "x1"], data_df.loc[~correct, "x2"], c="red", label="wrong")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Predictions: correct (green) vs wrong (red)")
    plt.legend()
    #plt.show()
    
    # Consider, in addition, the product of x1 and x2 as predictor along with the previous ones 
    x1x2 = x1 * x2 

    # create a new dataframe 
    data_df = pd.DataFrame(data={"x1":x1, "x2": x2, "x1x2": x1x2, "x1^2": x1_2, "x2^2": x2_2, "y": y})
    X = data_df[["x1", "x2", "x1x2", "x1^2", "x2^2"]]

    # Fit the logistic regression 
    clf = models.logistic_regression().fit(X, Y)
    Y_pred = clf.predict(X)

    # Confusion matrix of the results 
    m = confusion_matrix(Y, Y_pred)

    # Percentage of correct predictions 
    correct_rate = np.mean(Y_pred == Y)

    # Plot the predictions and color them according to the class label
    correct = (Y_pred == Y.to_numpy())
    
    plt.figure()
    plt.scatter(data_df.loc[correct, "x1"], data_df.loc[correct, "x2"], c="green", label="correct")
    plt.scatter(data_df.loc[~correct, "x1"], data_df.loc[~correct, "x2"], c="red", label="wrong")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Predictions: correct (green) vs wrong (red)")
    plt.legend()
    #plt.show()

    # Define a new pandas df
    data_df = pd.DataFrame(data={"x1":x1, "x2": x2, "y": y})
    X = data_df[["x1", "x2"]]
    
    # Fit the SVC 
    clf = models.svc(C=5, kernel="linear")
    clf.fit(X,Y)
    Y_pred = clf.predict(X)

    # Confusion matrix of the results 
    m = confusion_matrix(Y, Y_pred)

    # Percentage of correct predictions 
    correct_rate = np.mean(Y_pred == Y)
    
    # Perform CV to get a better value of C
    kfold = KFold(
        n_splits=5,
        random_state=42,
        shuffle=True
    )
    grid = GridSearchCV(
        clf,
        {'C': [0.001, 0.01, 0.1, 1, 5, 10, 100]},
        refit=True,
        cv=kfold,
        scoring='accuracy'
        )
    grid.fit(X, Y)

    # Fit the SVC with a polynomial kernel of degree 2
    clf = models.svc(
        C = 1,
        kernel='poly',
        degree = 2
    )
    clf.fit(X, Y)
    Y_pred = clf.predict(X)

    # Confusion matrix of the results 
    m = confusion_matrix(Y, Y_pred)

    # Percentage of correct predictions 
    correct_rate = np.mean(Y_pred == Y)

if __name__ == "__main__":
    main()
