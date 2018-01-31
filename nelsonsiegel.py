import numpy as np
import scipy as sc
import statsmodels.api as sm
import pandas as pd

def nsrates(coeff, maturity = np.array([2,5,10,20]), colname = "NSrate_"):
    result = pd.DataFrame()
    for mat in maturity:
        result[colname + str(mat)] = coeff['beta0'] + coeff['beta1'].multiply(factorBeta1(coeff['lambda'], mat)) + coeff['beta2'].multiply(factorBeta2(coeff['lambda'], mat))
    return result

def nsrates_interval(coeff, interval = (2,20), smoothness = 0.5, colname = "NSrate_"):
    maturity = np.arange(interval[0], interval[1]+smoothness, smoothness)
    result = pd.DataFrame(columns = maturity)
    for mat in maturity:
        result[mat] = coeff['beta0'] + coeff['beta1'].multiply(factorBeta1(coeff['lambda'], mat)) + coeff['beta2'].multiply(factorBeta2(coeff['lambda'], mat))
    return result


def beta1Spot(maturity, tau):
    return np.divide((1 - np.exp(-np.divide(maturity, tau))), (np.divide(maturity, tau)))

def beta2Spot(maturity, tau):
    return (np.divide((1 - np.exp(-np.divide(maturity, tau))), (np.divide(maturity, tau))) - np.exp(-np.divide(maturity, tau)))

def beta1Forward(maturity, tau):
    return np.exp(-np.divide(maturity, tau))

def beta2Forward(maturity, tau):
    return np.multiply(np.exp(-np.divide(maturity, tau)), (np.divide(maturity, tau)))


def factorBeta1(lambda_var, maturity):
    return np.divide((1-np.exp(-np.multiply(lambda_var, maturity))), (np.multiply(lambda_var, maturity)))

def factorBeta2(lambda_var, maturity):
    return np.divide((1-np.exp(-np.multiply(lambda_var, maturity))), (np.multiply(lambda_var, maturity))) - np.exp(-np.multiply(lambda_var, maturity))


def factorBeta2neg(lambda_var, maturity):
    return np.negative(np.divide((1-np.exp(-np.multiply(lambda_var, maturity))), (np.multiply(lambda_var, maturity))) - np.exp(-np.multiply(lambda_var, maturity)))

def ns_estimator(rate, maturity, lambda_var):
    model = sm.OLS( rate.values.reshape(len(rate),1), sm.add_constant(
        np.column_stack(
            (factorBeta1(lambda_var,maturity),
             factorBeta2(lambda_var,maturity)))))
    beta = model.fit()
    betaPar = beta.params
    naValues = betaPar[~np.isnan(betaPar)]
    if len(naValues) < 3:
        betaPar = np.array([0, 0, 0])
    #pd.DataFrame(data = np.reshape(betaPar, ( 1, len(betaPar))), columns = ["beta0", "beta1", "beta2"])
    estResults = {'Par': betaPar, 'Res': beta.resid}
    return estResults

def nss_estimator(rate, maturity, tau1, tau2):
    model = sm.OLS( rate.values.reshape(1, len(rate)), sm.add_constant(np.column_stack((beta1Spot(maturity,tau1),
                                                          beta2Spot(maturity,tau1),
                                                          beta2Spot(maturity,tau2)
                                                         ))))
    beta = model.fit()
    betaPar = beta.params
    naValues = betaPar[~np.isnan(betaPar)]
    if len(naValues) < 4:
        betaPar = np.array([0, 0, 0, 0])
    pd.DataFrame(data = betaPar, columns = ["beta0", "beta1", "beta2","beta3"])
    estResults = {'Par': betaPar, 'Res': beta.resid}
    return estResults



def nelson_siegel(rate, maturity):
    maturity = np.array(maturity)
    pillars_number = len(maturity) - 1
    lambdaValues = np.arange(maturity[0], maturity[ pillars_number ], 0.5)

    finalResults = pd.DataFrame(data = np.zeros((len(rate), 4)),
                                columns = ["beta0","beta1","beta2","lambda"])
    j = 0
    success_n = 0
    error_n = 0
    while j < len(rate):
        interResults = pd.DataFrame(data = np.zeros((len(lambdaValues), 5)),
                                columns = ["beta0","beta1","beta2","lambda","SSR"])
        for i in range(0, len(lambdaValues)):
            lambdaTemp = sc.optimize.minimize(factorBeta2neg, x0 = 1, args=(lambdaValues[i],)).x
            interEstimation = ns_estimator(rate.iloc[j], maturity, lambdaTemp)
            try:
                betaCoef = interEstimation["Par"].as_matrix()
                success_n += 1
            except:
                betaCoef = interEstimation["Par"]
                error_n += 1
            if betaCoef[0] > 0 and betaCoef[0] < 20:
                SSR = np.sum(np.power(interEstimation["Res"], 2))
                row = np.append(np.append(betaCoef, lambdaTemp), SSR)
                interResults.iloc[i] = row
            else:
                row = np.append(np.append(betaCoef, lambdaValues[i]), 1e+5)
                interResults.iloc[i] = row
        BestRow = interResults.iloc[:, 4].argmin()
        finalResults.iloc[j] = interResults.iloc[BestRow,0:4]
        j += 1
    finalResults.index = rate.index
    return finalResults
