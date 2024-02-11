import pandas as pd
import Lab02_solution as lb
import fun
import pprint
import numpy as np
from scipy.optimize import minimize, least_squares 

maturities = [1,2,3,5,10]

germany_1 = pd.read_csv('Bond/Germany 1-Year Bond Yield Historical Data.csv')
germany_2 = pd.read_csv('Bond/Germany 2-Year Bond Yield Historical Data.csv')
germany_3 = pd.read_csv('Bond/Germany 3-Year Bond Yield Historical Data.csv')
germany_5 = pd.read_csv('Bond/Germany 5-Year Bond Yield Historical Data.csv')
germany_10 = pd.read_csv('Bond/Germany 10-Year Bond Yield Historical Data.csv')

portugal_1 = pd.read_csv('Bond/Portugal 1-Year Bond Yield Historical Data.csv')
portugal_2 = pd.read_csv('Bond/Portugal 2-Year Bond Yield Historical Data.csv')
portugal_3 = pd.read_csv('Bond/Portugal 3-Year Bond Yield Historical Data.csv')
portugal_5 = pd.read_csv('Bond/Portugal 5-Year Bond Yield Historical Data.csv')
portugal_10 = pd.read_csv('Bond/Portugal 10-Year Bond Yield Historical Data.csv')

sk_1 = pd.read_csv('Bond/South Korea 1-Year Bond Yield Historical Data.csv')
sk_2 = pd.read_csv('Bond/South Korea 2-Year Bond Yield Historical Data.csv')
sk_3 = pd.read_csv('Bond/South Korea 3-Year Bond Yield Historical Data.csv')
sk_5 = pd.read_csv('Bond/South Korea 5-Year Bond Yield Historical Data.csv')
sk_10 = pd.read_csv('Bond/South Korea 10-Year Bond Yield Historical Data.csv')

usa_1 = pd.read_csv('Bond/United States 1-Year Bond Yield Historical Data.csv')
usa_2 = pd.read_csv('Bond/United States 2-Year Bond Yield Historical Data.csv')
usa_3 = pd.read_csv('Bond/United States 3-Year Bond Yield Historical Data.csv')
usa_5 = pd.read_csv('Bond/United States 5-Year Bond Yield Historical Data.csv')
usa_10 = pd.read_csv('Bond/United States 10-Year Bond Yield Historical Data.csv')

# Germany
germany_1_df = pd.DataFrame(germany_1)
germany_2_df = pd.DataFrame(germany_2)
germany_3_df = pd.DataFrame(germany_3)
germany_5_df = pd.DataFrame(germany_5)
germany_10_df = pd.DataFrame(germany_10)

germany_1_df_clear = fun.clear_df(germany_1_df)
germany_2_df_clear = fun.clear_df(germany_2_df)
germany_3_df_clear = fun.clear_df(germany_3_df)
germany_5_df_clear = fun.clear_df(germany_5_df)
germany_10_df_clear = fun.clear_df(germany_10_df)
all_df_joint_germany = fun.join_df_date(germany_1_df_clear, germany_2_df_clear, germany_3_df_clear, germany_5_df_clear, germany_10_df_clear, 1, 2, 3, 5, 10)

# Portugal
portugal_1_df = pd.DataFrame(portugal_1)
portugal_2_df = pd.DataFrame(portugal_2)
portugal_3_df = pd.DataFrame(portugal_3)
portugal_5_df = pd.DataFrame(portugal_5)
portugal_10_df = pd.DataFrame(portugal_10)

portugal_1_df_clear = fun.clear_df(portugal_1_df)
portugal_2_df_clear = fun.clear_df(portugal_2_df)
portugal_3_df_clear = fun.clear_df(portugal_3_df)
portugal_5_df_clear = fun.clear_df(portugal_5_df)
portugal_10_df_clear = fun.clear_df(portugal_10_df)
all_df_joint_portugal = fun.join_df_date(portugal_1_df_clear, portugal_2_df_clear, portugal_3_df_clear, portugal_5_df_clear, portugal_10_df_clear, 1, 2, 3, 5, 10)

# South Korea
sk_1_df = pd.DataFrame(sk_1)
sk_2_df = pd.DataFrame(sk_2)
sk_3_df = pd.DataFrame(sk_3)
sk_5_df = pd.DataFrame(sk_5)
sk_10_df = pd.DataFrame(sk_10)

sk_1_df_clear = fun.clear_df(sk_1_df)
sk_2_df_clear = fun.clear_df(sk_2_df)
sk_3_df_clear = fun.clear_df(sk_3_df)
sk_5_df_clear = fun.clear_df(sk_5_df)
sk_10_df_clear = fun.clear_df(sk_10_df)
all_df_joint_sk = fun.join_df_date(sk_1_df_clear, sk_2_df_clear, sk_3_df_clear, sk_5_df_clear, sk_10_df_clear, 1, 2, 3, 5, 10)

# US
us_1_df = pd.DataFrame(usa_1)
us_2_df = pd.DataFrame(usa_2)
us_3_df = pd.DataFrame(usa_3)
us_5_df = pd.DataFrame(usa_5)
us_10_df = pd.DataFrame(usa_10)

us_1_df_clear = fun.clear_df(us_1_df)
us_2_df_clear = fun.clear_df(us_2_df)
us_3_df_clear = fun.clear_df(us_3_df)
us_5_df_clear = fun.clear_df(us_5_df)
us_10_df_clear = fun.clear_df(us_10_df)
all_df_joint_us = fun.join_df_date(us_1_df_clear, us_2_df_clear, us_3_df_clear, us_5_df_clear, us_10_df_clear, 1, 2, 3, 5, 10)

dates = list(us_1_df_clear.index)

all_df_joint = {'Germany': all_df_joint_germany, 
                'Portugal': all_df_joint_portugal,
                'South Korea': all_df_joint_sk,
                'United States': all_df_joint_us}

# Parameters
beta0 = 0.01
beta1 = 0.05
beta2 = 0.2
beta3 = 0.03
tau = 1
tau2 = 1

params_NS = [beta0, beta1, beta2, tau]
params_NSS = [beta0, beta1, beta2, beta3, tau, tau2]

#Parameters for Gradient Descent
alpha_0 = 1
apx_LS = False
N = 100


for name_country, df_joint_country in all_df_joint.items():
    # Compute R
    for df in df_joint_country:
        df['Nelson-Siegel'] = fun.compute_R(df['Maturity'], params_NS=params_NS)
        df['Nelson-Siegel-Svensson'] = fun.compute_R(df['Maturity'], params_NSS=params_NSS)
        
    # Loop over each dataframe in all_df_joint_germany
    params_values_list = []
    f_values_list = []
    params_values_list_NSS = []
    f_values_list_NSS = []

    # Loop over each dataframe in 
    for df in df_joint_country:
    
        # Compute f and minimize using gradient descent
        params_values, f_values = lb.gradient_descent(lambda params: fun.compute_f(df['Yield'], df['Maturity'], params_NS=params),
                                                       params_NS, alpha_0=alpha_0, apx_LS=apx_LS, N = N)
        params_values_NSS, f_values_NSS = lb.gradient_descent(lambda params: fun.compute_f(df['Yield'], df['Maturity'], params_NSS = params),
                                                               params_NSS, alpha_0=alpha_0, apx_LS=apx_LS, N = N)

        # Append results to lists
        params_values_list.append(params_values)
        f_values_list.append(f_values)
        params_values_list_NSS.append(params_values_NSS)
        f_values_list_NSS.append(f_values_NSS)

    # Convert lists to dataframes
    df_params = pd.DataFrame(params_values_list)
    df_f = pd.DataFrame(f_values_list)
    df_params_NSS = pd.DataFrame(params_values_list_NSS)
    df_f_NSS = pd.DataFrame(f_values_list_NSS)

    fun.excel(params_values_list, name_country, 'Nelson-Siegel', 'Gradient Descent', 'Parameters')
    fun.excel(f_values_list, name_country, 'Nelson-Siegel', 'Gradient Descent', 'Function')
    fun.excel(params_values_list_NSS, name_country, 'Nelson-Siegel-Svensson', 'Gradient Descent', 'Parameters')
    fun.excel(f_values_list_NSS, name_country, 'Nelson-Siegel-Svensson', 'Gradient Descent', 'Function')

    # Paste Optimal Values into DFs and Plot them
    for index, df in enumerate(df_joint_country):

        # Extract parameter values from df_params
        beta0 = df_params.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][0]
        beta1 = df_params.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][1]
        beta2 = df_params.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][2]
        tau = df_params.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][3]
    
        beta0_NSS = df_params_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][0]
        beta1_NSS = df_params_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][1]
        beta2_NSS = df_params_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][2]
        beta3_NSS = df_params_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][3]
        tau_NSS = df_params_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][4]
        tau2_NSS = df_params_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][5]
    
        # Create params_NS list
        params_NS_optimal = [beta0, beta1, beta2, tau]
        params_NSS_optimal = [beta0_NSS, beta1_NSS, beta2_NSS, beta3_NSS, tau_NSS, tau2_NSS]

        # Compute Nelson-Siegel curve values
        df['Nelson-Siegel'] = fun.compute_R(df['Maturity'], params_NS=params_NS_optimal)
        df['Nelson-Siegel-Svenson'] = fun.compute_R(df['Maturity'], params_NSS=params_NSS_optimal)

        # Plot the curve
        fun.plot_curve(maturities, df['Yield'], df['Nelson-Siegel'], name_country, 'Nelson-Siegel', 'Gradient Descent', dates[index])
        fun.plot_curve(maturities, df['Yield'], df['Nelson-Siegel-Svenson'], name_country, 'Nelson-Siegel-Svensson',
                        'Gradient Descent', dates[index])
    
    # Newton Method
    params_values_newton_NS = []
    f_values_newton_NS = []
    params_values_newton_NSS = []
    f_values_newton_NSS = []

    # Loop over each dataframe in all_df_joint_germany
    for df in df_joint_country:
    
        # Compute f and minimize using gradient descent
        params_values, f_values = lb.newton_method(lambda params: fun.compute_f(df['Yield'], df['Maturity'], params_NS=params),
                                                    params_NS, N = N)
        params_values_NSS, f_values_NSS = lb.newton_method(lambda params: fun.compute_f(df['Yield'], df['Maturity'], params_NSS=params),
                                                            params_NSS, N = N)
    
        # Append results to lists
        params_values_newton_NS.append(params_values)
        f_values_newton_NS.append(f_values)
        params_values_newton_NSS.append(params_values_NSS)
        f_values_newton_NSS.append(f_values_NSS)

    # Convert lists to dataframes
    df_params_newton = pd.DataFrame(params_values_newton_NS)
    df_f_newton = pd.DataFrame(f_values_newton_NS)
    df_params_newton_NSS = pd.DataFrame(params_values_newton_NSS)
    df_f_newton_NSS = pd.DataFrame(f_values_newton_NSS)

    # Save everything in Excel
    fun.excel(params_values_newton_NS, name_country, 'Nelson-Siegel', 'Newton', 'Parameters')
    fun.excel(f_values_newton_NS, name_country, 'Nelson-Siegel', 'Newton', 'Function')
    fun.excel(params_values_list_NSS, name_country, 'Nelson-Siegel-Svensson', 'Newton', 'Parameters')
    fun.excel(f_values_list_NSS, name_country, 'Nelson-Siegel-Svensson', 'Newton', 'Function')


    # Paste Optimal for Newton and plot it
    for index, df in enumerate(df_joint_country):
    
        # Extract parameter values from df_params
        beta0 = df_params_newton.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][0]
        beta1 = df_params_newton.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][1]
        beta2 = df_params_newton.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][2]
        tau = df_params_newton.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][3]
        
        beta0_NSS = df_params_newton_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][0]
        beta1_NSS = df_params_newton_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][1]
        beta2_NSS = df_params_newton_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][2]
        beta3_NSS = df_params_newton_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][3]
        tau_NSS = df_params_newton_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][4]
        tau2_NSS = df_params_newton_NSS.iloc[:, 1:].ffill(axis=1).iloc[:, -1][index][5]
            
        # Create params_NS list
        params_NS_optimal = [beta0, beta1, beta2, tau]
        params_NSS_optimal = [beta0_NSS, beta1_NSS, beta2_NSS, beta3_NSS, tau_NSS, tau2_NSS]

        # Compute Nelson-Siegel curve values
        df['Nelson-Siegel'] = fun.compute_R(df['Maturity'], params_NS=params_NS_optimal)
        df['Nelson-Siegel-Svenson'] = fun.compute_R(df['Maturity'], params_NSS=params_NSS_optimal)

        # Plot the curve
        fun.plot_curve(maturities, df['Yield'], df['Nelson-Siegel'], name_country, 'Nelson-Siegel', 'Newton', dates[index])
        fun.plot_curve(maturities, df['Yield'], df['Nelson-Siegel-Svenson'], name_country, 'Nelson-Siegel-Svensson', 'Newton', dates[index])

    # BFGS method
    for index, df in enumerate(df_joint_country):
        results_NS = minimize(lambda params: fun.compute_f(df['Yield'], df['Maturity'], params_NS=params), params_NS, method='BFGS')
        results_NSS = minimize(lambda params: fun.compute_f(df['Yield'], df['Maturity'], params_NSS=params), params_NSS, method='BFGS')
        params_NS_BFGS = results_NS.x
        params_NSS_BFGS = results_NSS.x
        df['Nelson-Siegel'] = fun.compute_R(df['Maturity'], params_NS=params_NS_BFGS)
        df['Nelson-Siegel-Svensson'] = fun.compute_R(df['Maturity'], params_NSS=params_NSS_BFGS)
        fun.plot_curve(maturities, df['Yield'], df['Nelson-Siegel'], name_country, 'Nelson-Siegel', 'BFGS', dates[index])
        fun.plot_curve(maturities, df['Yield'], df['Nelson-Siegel-Svensson'], name_country, 'Nelson-Siegel-Svensson', 'BFGS', dates[index])

    # Levenberg-Marquardt method
    #results1 = least_squares(lambda params: fun.compute_f(df['Yield'], df['Maturity'], params_NS=params), params_NS, method='lm')
