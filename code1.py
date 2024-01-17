# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 09:25:36 2023

@author: Alessandro
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy as sp
import seaborn as sns
from statsmodels . tsa . stattools import adfuller

#creating 2 dataframes with datas
df =pd.read_excel ('monthly prices equities.xlsx ', index_col =None ,na_values =['NA'])
dfd =pd.read_excel ('daily prices equities.xlsx ', index_col =None ,na_values =['NA'])

#creating times

t=pd.date_range ( start ='30/11/2013',end ='30/11/2023 ', freq ='M')
tD= dfd[[ 'Exchange Date']]

#creating a list with company names for the for loop plotting
dataaaaaaaq =pd.read_excel('monthly prices equities.xlsx ',header=None, index_col =None ,na_values =['NA'])
company_names = data.iloc[0, 1:].tolist()
#plotting levels and log-levels for every equity monthly
for i in company_names:
    Equity= df[[i]]
    n=np.size ( Equity )
    plt . plot (t , Equity[::-1])
    
    plt . xlabel ('Time - Monthly - 30/11/2013- 30/11/2023 ')
    plt . ylabel (i)
    plt.show()
    plt.close()

#plotting the log-plots
# compute log - levels
    p=np . log (Equity)
    plt.semilogy(t, Equity[::-1])
    plt . xlabel ('logarithmic - Monthly - 30/11/2013- 30/11/2023 ')
    plt . ylabel (i)
    plt.show()
    plt.close()
    # compute returns
    rm= 100 *(p -p. shift (1) )
    #doing analysis of statystic indicators
    print("monthly",rm.describe()) 
    print("skewness of:",rm.skew())
    print("kurtosis of:",rm. kurt())
    #doing jarque bera test
    nm=np . size ( rm )
    print("Jarque bera",sp . stats . jarque_bera (rm[1:nm]))
    print()
    #plotting histogram
        
    plt . hist ( rm , bins =100 , density = True )
    plt . xlabel ('Time - Monthly - 30/11/2013- 30/11/2023 ')
    plt . ylabel (i)
    plt.show()
    plt.close()
 
#plotting levels and log-levels for every equity daily
for i in company_names:
    Equity= dfd[[i]]
    n=np.size ( Equity )
    plt . plot (tD , Equity)
    plt . xlabel ('Time - Daily - 20/11/2013- 30/11/2023 ')
    plt . ylabel (i)
    plt.show()
    plt.close()


#plotting the log-plots
# compute log - levels
    p=np . log (Equity)
    plt.semilogy(tD, Equity)
    plt . xlabel ('logarithmic - Daily - 20/11/2013- 30/11/2023 ')
    plt . ylabel (i)
    plt.show()
    plt.close()
    # compute returns
    rd= 100 *(p -p. shift (1) )
    #doing analysis of statystic indicators
    print("daily",rd.describe()) 
    print("skewness of:",rd.skew())
    print("kurtosis of:",rd. kurt())
    #doing jarque bera test
    nd=np . size ( rd )
    print("Jarque bera",sp . stats . jarque_bera (rd[1:nd]))
    print()
    #plotting histogram
        
    plt . hist ( rd , bins =100 , density = True )
    plt . xlabel ('Time - Monthly - 30/11/2013- 30/11/2023 ')
    plt . ylabel (i)
    plt.show()
    plt.close()
    
"""
EXERCISE 3 ADF
"""
    
#cutting last 2 years for monthly
df2y = df.drop(index=df.index[:24]) #deleting first 24 rows of 2023 and 2022

#cutting last 6 months for daily
df6m = dfd.drop(index=df.index[:129]) #deleting first 129 rows of last 6 months

#ADF for every company monthly
for i in company_names:
    Equity= df2y[[i]]
    adftest_monthly= adfuller(Equity, maxlag = 11)
    print("ADF Test Statistic monthly for",i,":",adftest_monthly[0])
    print("p-value for",i,":", adftest_monthly[1])
    print("used lag for",i,":", adftest_monthly[2])
    print()
    # Take the first difference
    diff_data = Equity -Equity. shift (1)
    diff_data=diff_data.dropna()
    adftest_monthly= adfuller(diff_data, maxlag = 11)
    print("ADF Test Statistic first difference monthly for",i,":",adftest_monthly[0])
    print("p-value for",i,":", adftest_monthly[1])
    print("used lag for",i,":", adftest_monthly[2])
    print()
    
#ADF for every company daily
for i in company_names:
    Equity= df6m[[i]]
    adftest_daily = adfuller (Equity, maxlag = 21)
    print("ADF Test Statistic daily for",i,adftest_daily[0])
    print("p-value for",i, adftest_daily[1])
    print("used lag for",i,adftest_daily[2])
    print()
    # Take the first difference
    diff_data = Equity-Equity.shift(1)
    diff_data=diff_data.dropna()
    adftest_monthly= adfuller(diff_data, maxlag = 21)
    print("ADF Test Statistic first difference daily for",i,":",adftest_monthly[0])
    print("p-value for",i,":", adftest_monthly[1])
    print("used lag for",i,":", adftest_monthly[2])
    print()