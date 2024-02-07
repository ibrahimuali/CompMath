import pandas as pd
import Lab02_solution as lb
import fun
import pprint
import numpy as np

bond_yields = {}

countries = ['Germany', 'Portugal']
years = [1, 2, 3, 5, 7, 10]

for country in countries:
    for year in years:
        file_path = f'Bond/{country} {year}-Year Bond Yield Historical Data.csv'
        df = pd.read_csv(file_path)
        df_clear = fun.clear_df(pd.DataFrame(df))
        bond_yields[f'{country}_{year}'] = df_clear

all_df_joint = {}
for country in countries:
    dfs = [bond_yields[f'{country}_{year}'] for year in years]
    all_df_joint[country] = fun.join_df_date(*dfs, *years)

pprint.pprint(all_df_joint)