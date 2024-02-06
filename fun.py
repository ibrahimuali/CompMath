import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def clear_df(df):
    '''
    Drop useless columns and set date as index

    Return: DataFrame
    '''

    columns_to_drop = ['Open', 'High', 'Low', 'Change %']
    df.drop(columns=columns_to_drop, inplace=True)
    df.set_index('Date', inplace=True)
    df['Price'] = (df['Price']/100)
    df = df[::-1]
    return df

def join_df_date(df1, df2, df3, df4, df5, df6, maturity1, maturity2, maturity3, maturity4, maturity5, maturity6):
    '''
    Join two Dataframe with a column for Price and other one for Maturity for each month.
    
    Return: 
    - all_df: list
        It is a list of Dataframe of each month
    '''
    all_df = []
    data = {'Maturity': [maturity1, maturity2, maturity3, maturity4, maturity5, maturity6]}
    for i in range(len(df1)):
        price = {'Yield': [df1['Price'].iloc[i], df2['Price'].iloc[i], df3['Price'].iloc[i], df4['Price'].iloc[i], df5['Price'].iloc[i], df6['Price'].iloc[i]]}
        df_new = pd.DataFrame({**data, **price})
        all_df.append(df_new)
    return all_df

def compute_R(tau, params_NS=None, params_NSS=None, tau2=None):
    """
    Computes the R(t) value using the Nelson-Siegel model.

    Parameters:
    - time: column of dataframe
        The time parameter for which R(t) is to be computed.
    - params: array
        Parameters of the Nelson-Siegel model
    
    Returns:
    - float:
        The computed R(t) value.
    """
    if params_NS is not None:
        beta0, beta1, beta2, time = params_NS
        f1 = (1 - np.exp(-time / tau)) / (time / tau)
        f2 = (1 - np.exp(-time / tau)) / (time / tau) - np.exp(-time / tau)
        return beta0 + beta1 * f1 + beta2 * f2
    else:
        beta0, beta1, beta2, beta3, time = params_NSS
        f3 = (1 - np.exp(-time / tau2)) / (time / tau2) - np.exp(-time / tau2)
        return beta0 + beta1 * f1 + beta2 * f2 + beta3 * f3
    

def compute_f(yields, tau, params_NS=None, params_NSS=None, tau2=None):
        """
        Computes the f(β0, β1, β2, τ) or f(β0, β1, β2, β3, τ_0, τ_1) value using the Nelson-Siegel model.

        Returns:
        - float:
            The computed f(β0, β1, β2, τ) or f(β0, β1, β2, β3, τ_0, τ_1) value.
        """
        if params_NS is not None:
            residuals = yields - compute_R(tau, params_NS=params_NS)
        else:
             residuals = yields - compute_R(tau, params_NSS=params_NSS, tau2=tau2)
        return np.sum(residuals**2)
    
def plot_curve(time, yields, R, method = '', save_folder=''):
        """
        Plots the curve (t, R(t)) using the computed R(t) values and the historical yield data.
        """
        if method == 'Newton':
            plt.plot(time, yields, label='Historical Yield Data')
            plt.plot(time, R, label='Nelson-Siegel Curve [Newton]')
            plt.xlabel('Time')
            plt.ylabel('Yield')
            plt.title('Nelson-Siegel Curve vs Historical Yield Data [Newton]')
            plt.legend()
            #if save_folder:
                # Save the plot to the specified folder
               # plt.savefig(os.path.join(save_folder, 'plot_newton.png'))
            #else:
            plt.show()
        else: 
            plt.plot(time, yields, label='Historical Yield Data')
            plt.plot(time, R, label='Nelson-Siegel Curve [Gradient Descent]')
            plt.xlabel('Time')
            plt.ylabel('Yield')
            plt.title('Nelson-Siegel Curve vs Historical Yield Data [Gradient Descent]')
            plt.legend()
            plt.show()
           # if save_folder:
                # Save the plot to the specified folder
              #  plt.savefig(os.path.join(save_folder, 'plot_Gradient Descent.png'))
            #else:
            plt.show()
