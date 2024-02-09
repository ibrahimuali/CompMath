import pandas as pd
import Lab02_solution as lb
import fun
import pprint
import numpy as np

bond_yields = {}

countries = ['Germany', 'Portugal', 'South Korea', 'United States']
years = [1, 2, 3, 5, 7, 10]
years_countries = []

for country in countries:
    for year in years:
        try:
            file_path = f'Bond/{country} {year}-Year Bond Yield Historical Data.csv'
            df = pd.read_csv(file_path)
            df_clear = fun.clear_df(pd.DataFrame(df))
            bond_yields[f'{country}_{year}'] = df_clear
            years_countries.append(f'{country}_{year}')
        except (KeyError, FileNotFoundError):
            continue

print(years_countries)

all_df_joint = {}
for country in countries:
    try:
            #dfs = [bond_yields[years_countries] for year in years]
            all_df_joint[country] = fun.join_df_date(*years_countries, *years)
    except KeyError:
            continue

# Parameters
beta0, beta1, beta2, beta3, tau, tau2 = 0.01, 0.05, 0.2, 0.03, 1, 1

params_NS = [beta0, beta1, beta2, tau]

# Parameters for Gradient Descent
alpha_0 = 1
apx_LS = True
N = 50

print(all_df_joint.keys())
# Compute R
for country in countries:
    for i in range(len(all_df_joint[country])):    
        all_df_joint[country][i]['Nelson-Siegel'] = fun.compute_R(all_df_joint[country][i]['Maturity'], params_NS=params_NS)
