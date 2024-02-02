import pandas as pd
import Lab02_solution as lb
import fun
import pprint
import numpy as np
import maths as ma

germany_1 = pd.read_csv('Bond/Germany 1-Year Bond Yield Historical Data.csv')
germany_3 = pd.read_csv('Bond/Germany 3-Year Bond Yield Historical Data.csv')
germany_5 = pd.read_csv('Bond/Germany 5-Year Bond Yield Historical Data.csv')
germany_10 = pd.read_csv('Bond/Germany 10-Year Bond Yield Historical Data.csv')

usa_1 = pd.read_csv('Bond/United States 1-Year Bond Yield Historical Data.csv')
usa_5 = pd.read_csv('Bond/United States 5-Year Bond Yield Historical Data.csv')

portugal_1 = pd.read_csv('Bond/Portugal 1-Year Bond Yield Historical Data.csv')
portugal_5 = pd.read_csv('Bond/Portugal 5-Year Bond Yield Historical Data.csv')

germany_1_df = pd.DataFrame(germany_1)
germany_3_df = pd.DataFrame(germany_3)
germany_5_df = pd.DataFrame(germany_5)
germany_10_df = pd.DataFrame(germany_10)

portugal_1_df =pd.DataFrame(portugal_1)
portugal_5_df = pd.DataFrame(portugal_5)
    
# Germany
germany_1_df_clear = fun.clear_df(germany_1_df)
germany_3_df_clear = fun.clear_df(germany_3_df)
germany_5_df_clear = fun.clear_df(germany_5_df)
germany_10_df_clear = fun.clear_df(germany_10_df)
all_df_joint_germany = fun.join_df_date(germany_1_df_clear, germany_3_df_clear, germany_5_df_clear, germany_10_df_clear, 1, 3, 5, 10)

# Portugal
portugal_1_df_clear = fun.clear_df(portugal_1_df)
portugal_5_df_clear = fun.clear_df(portugal_5_df)
#all_df_joint_portugal = fun.join_df_date(portugal_1_df_clear, portugal_5_df_clear, 1, 5)

# Parameters
beta0 = 0.01
beta1 = 0.01
beta2 = 0.01
beta3 = 0.01
tau = 1
tau2 = 1

params_NS = [beta0, beta1, beta2, tau]

# Compute R
all_df_joint_germany[20]['Nelson-Siegel'] = fun.compute_R(all_df_joint_germany[20]['Maturity'], params_NS=params_NS)
all_df_joint_germany[20].to_excel('data.xlsx')

# Compute f and minimize
params_values, f_values = lb.gradient_descent(lambda params: fun.compute_f(all_df_joint_germany[20]['Yield'], all_df_joint_germany[20]['Maturity'], params_NS=params), params_NS)
df_params = pd.DataFrame(params_values)
df_f = pd.DataFrame(f_values)
with pd.ExcelWriter('ParametersValues.xlsx') as writer:
    df_params.to_excel(writer, sheet_name='Params', index=False)
    df_f.to_excel(writer, sheet_name='F_values', index=False)
