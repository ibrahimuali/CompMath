import pandas as pd
import Lab02_solution as lb
import fun
import numpy as np
from scipy.optimize import minimize, least_squares 

bond_yields = {}

countries = ['Germany', 'Portugal', 'South Korea', 'United States']
maturities = [1, 2, 3, 5, 10]
countries_years = []

for country in countries:
    for year in maturities:
        try:
            file_path = f'Bond/{country} {year}-Year Bond Yield Historical Data.csv'
            df = pd.read_csv(file_path)
            df_clear = fun.clear_df(pd.DataFrame(df))
            bond_yields[f'{country}_{year}'] = df_clear
            countries_years.append(f'{country}_{year}')
        except (KeyError, FileNotFoundError):
            continue
        print(countries_years)
'''all_df_joint = {}
for country in countries:
    try:
        dfs = [bond_yields[year] for year in countries_years]
        all_df_joint[country] = fun.join_df_date(dfs, maturities)
    except KeyError:
        continue

# Parameters
beta0, beta1, beta2, beta3, tau, tau2 = 0.01, 0.05, 0.2, 0.03, 1, 1

params_NS = [beta0, beta1, beta2, tau]
params_NSS = [beta0, beta1, beta2, beta3, tau, tau2]

# Parameters for Gradient Descent
alpha_0 = 1
apx_LS = True
N = 2

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
        fun.plot_curve(maturities, df['Yield'], df['Nelson-Siegel'], 'US', 'Nelson-Siegel', 'Newton', dates[index])
        fun.plot_curve(maturities, df['Yield'], df['Nelson-Siegel-Svenson'], 'US', 'Nelson-Siegel-Svensson', 'Newton', dates[index])

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

'''