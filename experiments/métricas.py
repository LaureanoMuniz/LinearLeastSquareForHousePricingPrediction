import numpy as np


def rmse(actual, predicted):
    return np.sqrt(np.mean((actual-predicted)**2))


def rmsle(actual, predicted):
    return np.sqrt(
        np.mean(
            (np.log(actual+1)-np.log(predicted+1))**2
        )
    )


def cp(actual, predicted, p, predicted_all):
    n = len(actual)
    sse = ((actual-predicted)**2).sum()
    s2 = ((actual - predicted_all)**2).sum()/n
    return sse/s2 - n + 2*(p+1)


def r2(actual, predicted):
    m = np.mean(actual)
    ss_tot = ((actual - m)**2).sum()
    ss_res = ((actual - predicted)**2).sum()
    return 1 - ss_res/ss_tot


def r2_adjusted(actual, predicted, p):
    n = len(actual)
    return 1 - (1 - r2(actual, predicted)) * (n - 1) / (n - p - 1)

def precision_casas(actual, predicted):
    n = np.shape(actual)[0]
    out = 0
    for i in range(n):
        if(actual[i] <= 0.5 and predicted[i] <= 0.5):
            out += 1
        elif(actual[i] >= 0.5 and predicted[i] >= 0.5):
            out += 1
    return out/n