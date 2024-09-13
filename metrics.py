from sklearn.metrics import accuracy_score, recall_score, f1_score
import numpy as np

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred)

def f1_metric(y_true, y_pred):
    return f1_score(y_true, y_pred)

def specificity(y_true, y_pred):
    true_negative = np.sum((y_true == 0) & (y_pred == 0))
    false_positive = np.sum((y_true == 0) & (y_pred == 1))
    return true_negative / (true_negative + false_positive)
