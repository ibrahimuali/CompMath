import pandas as pd
import Lab02_solution as lb
import fun
import pprint
import numpy as np
import maths as ma

germany_1 = pd.read_csv('Bond/Germany 1-Year Bond Yield Historical Data.csv')
germany_2 = pd.read_csv('Bond/Germany 2-Year Bond Yield Historical Data.csv')
germany_3 = pd.read_csv('Bond/Germany 3-Year Bond Yield Historical Data.csv')
germany_5 = pd.read_csv('Bond/Germany 5-Year Bond Yield Historical Data.csv')
germany_7 = pd.read_csv('Bond/Germany 7-Year Bond Yield Historical Data.csv')
germany_10 = pd.read_csv('Bond/Germany 10-Year Bond Yield Historical Data.csv')

usa_1 = pd.read_csv('Bond/United States 1-Year Bond Yield Historical Data.csv')
usa_2 = pd.read_csv('Bond/United States 2-Year Bond Yield Historical Data.csv')
usa_3 = pd.read_csv('Bond/United States 3-Year Bond Yield Historical Data.csv')
usa_5 = pd.read_csv('Bond/United States 5-Year Bond Yield Historical Data.csv')
usa_7 = pd.read_csv('Bond/United States 7-Year Bond Yield Historical Data.csv')
usa_10 = pd.read_csv('Bond/United States 10-Year Bond Yield Historical Data.csv')

portugal_1 = pd.read_csv('Bond/Portugal 1-Year Bond Yield Historical Data.csv')
portugal_2 = pd.read_csv('Bond/Portugal 2-Year Bond Yield Historical Data.csv')
portugal_3 = pd.read_csv('Bond/Portugal 3-Year Bond Yield Historical Data.csv')
portugal_5 = pd.read_csv('Bond/Portugal 5-Year Bond Yield Historical Data.csv')
portugal_7 = pd.read_csv('Bond/Portugal 7-Year Bond Yield Historical Data.csv')
portugal_10 = pd.read_csv('Bond/Portugal 10-Year Bond Yield Historical Data.csv')

# Germany
germany_1_df = pd.DataFrame(germany_1)
germany_2_df = pd.DataFrame(germany_2)
germany_3_df = pd.DataFrame(germany_3)
germany_5_df = pd.DataFrame(germany_5)
germany_7_df = pd.DataFrame(germany_7)
germany_10_df = pd.DataFrame(germany_10)

germany_1_df_clear = fun.clear_df(germany_1_df)
germany_2_df_clear = fun.clear_df(germany_2_df)
germany_3_df_clear = fun.clear_df(germany_3_df)
germany_5_df_clear = fun.clear_df(germany_5_df)
germany_7_df_clear = fun.clear_df(germany_7_df)
germany_10_df_clear = fun.clear_df(germany_10_df)
all_df_joint_germany = fun.join_df_date(germany_1_df_clear, germany_2_df_clear, germany_3_df_clear, germany_5_df_clear, germany_7_df_clear, germany_10_df_clear, 1, 2, 3, 5, 7, 10)

# Portugal
portugal_1_df = pd.DataFrame(portugal_1)
portugal_2_df = pd.DataFrame(portugal_2)
portugal_3_df = pd.DataFrame(portugal_3)
portugal_5_df = pd.DataFrame(portugal_5)
portugal_7_df = pd.DataFrame(portugal_7)
portugal_10_df = pd.DataFrame(portugal_10)

portugal_1_df_clear = fun.clear_df(portugal_1_df)
portugal_2_df_clear = fun.clear_df(portugal_2_df)
portugal_3_df_clear = fun.clear_df(portugal_3_df)
portugal_5_df_clear = fun.clear_df(portugal_5_df)
portugal_7_df_clear = fun.clear_df(portugal_7_df)
portugal_10_df_clear = fun.clear_df(portugal_10_df)
all_df_joint_portugal = fun.join_df_date(portugal_1_df_clear, portugal_2_df_clear, portugal_3_df_clear, portugal_5_df_clear, portugal_7_df_clear, portugal_10_df_clear, 1, 2, 3, 5, 7, 10)

#US
us_1_df = pd.DataFrame(usa_1)
us_2_df = pd.DataFrame(usa_2)
us_3_df = pd.DataFrame(usa_3)
us_5_df = pd.DataFrame(usa_5)
us_7_df = pd.DataFrame(usa_7)
us_10_df = pd.DataFrame(usa_10)

us_1_df_clear = fun.clear_df(us_1_df)
us_2_df_clear = fun.clear_df(us_2_df)
us_3_df_clear = fun.clear_df(us_3_df)
us_5_df_clear = fun.clear_df(us_5_df)
us_7_df_clear = fun.clear_df(us_7_df)
us_10_df_clear = fun.clear_df(us_10_df)
all_df_joint_us = fun.join_df_date(us_1_df_clear, us_2_df_clear, us_3_df_clear, us_5_df_clear, us_7_df_clear, us_10_df_clear, 1, 2, 3, 5, 7, 10)

# Parameters
beta0 = 0.01
beta1 = 0.05
beta2 = 0.2
beta3 = 0.03
tau = 1
tau2 = 1

params_NS = [beta0, beta1, beta2, tau]

#Parameters for Gradient Descent
alpha_0 = 1
apx_LS = True
N = 50

# Compute R
#all_df_joint_germany[20]['Nelson-Siegel'] = fun.compute_R(all_df_joint_germany[20]['Maturity'], params_NS=params_NS)
#all_df_joint_germany[20].to_excel('data.xlsx')
for index, df in enumerate(all_df_joint_us):
    df['Nelson-Siegel'] = fun.compute_R(df['Maturity'], params_NS=params_NS)
    # Export the dataframe to Excel
    #df.to_excel(f'data_{index}.xlsx')

# Compute f and minimize
#params_values, f_values = lb.gradient_descent(lambda params: fun.compute_f(all_df_joint_germany[10]['Yield'], all_df_joint_germany[10]['Maturity'], params_NS=params), params_NS, alpha_0 = alpha_0, apx_LS = apx_LS, N = 100)
#df_params = pd.DataFrame(params_values)
#df_f = pd.DataFrame(f_values)

# Loop over each dataframe in all_df_joint_germany
params_values_list = []
f_values_list = []

# Loop over each dataframe in all_df_joint_germany
for index, df in enumerate(all_df_joint_us):
    # Compute f and minimize using gradient descent
    params_values, f_values = lb.gradient_descent(lambda params: fun.compute_f(df['Yield'], df['Maturity'], params_NS=params), params_NS, alpha_0=alpha_0, apx_LS=apx_LS, N = N)
    # Append results to lists
    params_values_list.append(params_values)
    f_values_list.append(f_values)

# Convert lists to dataframes
df_params = pd.DataFrame(params_values_list)
df_f = pd.DataFrame(f_values_list)

#Paste Optimal Values into DFs and Plot them
for index, df in enumerate(all_df_joint_us):
    # Extract parameter values from df_params
    beta0 = df_params[N][index][0]
    beta1 = df_params[N][index][1]
    beta2 = df_params[N][index][2]
    tau = df_params[N][index][3]
    # Create params_NS list
    params_NS_optimal = [beta0, beta1, beta2, tau]
    # Compute Nelson-Siegel curve values
    df['Nelson-Siegel'] = fun.compute_R(df['Maturity'], params_NS=params_NS_optimal)
    # Plot the curve
    time = range(1, len(df['Yield']) + 1)
    fun.plot_curve(time, df['Yield'], df['Nelson-Siegel'], method = 'Gradient Descent', save_folder = save_folder)
    
    
#Newton Method
params_values_newton = []
f_values_list_newton = []

# Loop over each dataframe in all_df_joint_germany
for index, df in enumerate(all_df_joint_us):
    # Compute f and minimize using gradient descent
    params_values, f_values = lb.newton_method(lambda params: fun.compute_f(df['Yield'], df['Maturity'], params_NS=params), params_NS, N = N)
    # Append results to lists
    params_values_newton.append(params_values)
    f_values_list_newton.append(f_values)

# Convert lists to dataframes
df_params_newton = pd.DataFrame(params_values_newton)
df_f_newton = pd.DataFrame(f_values_list_newton)

#Save everything in Excel
with pd.ExcelWriter('ParametersValues.xlsx') as writer:
    df_params.to_excel(writer, sheet_name='Params', index=False)
    df_f.to_excel(writer, sheet_name='F_values', index=False)
    df_params.to_excel(writer, sheet_name='Params_Newton', index=False)
    df_f.to_excel(writer, sheet_name='F_values_Newton', index=False)
    
#paste Optimal for Newton and plot it
for index, df in enumerate(all_df_joint_us):
    # Extract parameter values from df_params
    for i in range(len(df_params_newton) - 1, -1, -1):
        if not np.isnan(df_params_newton[i][index][0]):
            beta0 = df_params_newton[i][index][0]
            beta1 = df_params_newton[i][index][1]
            beta2 = df_params_newton[i][index][2]
            tau = df_params_newton[i][index][3]
            break
            
    # Create params_NS list
    params_NS_optimal = [beta0, beta1, beta2, tau]
    # Compute Nelson-Siegel curve values
    df['Nelson-Siegel'] = fun.compute_R(df['Maturity'], params_NS=params_NS_optimal)
    # Plot the curve
    time = range(1, len(df['Yield']) + 1)
    fun.plot_curve(time, df['Yield'], df['Nelson-Siegel'], method = 'Newton', save_folder = save_folder)
    