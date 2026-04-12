import pandas as pd
import numpy as np

# y predict and target must be numpy array

def Precision_energy_based(y_predict, y_target):
    assert y_predict.shape[0] == y_target.shape[0], 'samples number not match'
    assert y_predict.shape[1] == y_target.shape[1], 'number of appliances not match'
    denominator = y_predict.sum(axis = 0)
    numerator = np.minimum(y_predict, y_target).sum(axis = 0)
    return numerator / denominator
    # return a numpy array of precision score for appliances

def Recall_energy_based(y_predict, y_target):
    assert y_predict.shape[0] == y_target.shape[0], 'samples number not match'
    assert y_predict.shape[1] == y_target.shape[1], 'number of appliances not match'
    denominator = y_target.sum(axis = 0)
    numerator = np.minimum(y_predict, y_target).sum(axis = 0)
    return numerator / denominator

def F1_energy_based(y_predict, y_target):
    assert y_predict.shape[0] == y_target.shape[0], 'samples number not match'
    assert y_predict.shape[1] == y_target.shape[1], 'number of appliances not match'
    numerator = np.minimum(y_predict, y_target).sum(axis = 0)
    p_denominator = y_predict.sum(axis = 0)
    r_denominator = y_target.sum(axis = 0)
    precision = numerator / p_denominator
    recall = numerator / r_denominator
    return 2 * precision * recall / (precision + recall)

def NEP(y_predict, y_target):
    assert y_predict.shape[0] == y_target.shape[0], 'samples number not match'
    assert y_predict.shape[1] == y_target.shape[1], 'number of appliances not match'
    denominator = y_target.sum(axis = 0)
    numerator = np.abs(y_target - y_predict).sum(axis = 0)
    return numerator / denominator

def MAE(y_predict, y_target):
    assert y_predict.shape[0] == y_target.shape[0], 'samples number not match'
    assert y_predict.shape[1] == y_target.shape[1], 'number of appliances not match'
    numerator = np.abs(y_target - y_predict).sum(axis = 0)
    return numerator / y_target.shape[0]