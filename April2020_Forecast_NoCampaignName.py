# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 09:54:33 2020

@author: kevin.castillo
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

#uploading data
xls = pd.ExcelFile('Overall_Data_Mar9_2020_NoCampaignNames.xlsx')
campaign1 = pd.read_excel(xls, 'Campaign1', index_col ='Date', parse_dates = True)
campaign2 = pd.read_excel(xls, 'Campaign2', index_col ='Date', parse_dates = True)
campaign3 = pd.read_excel(xls,'Campaign3', index_col ='Date', 
                         parse_dates = True)
campaign4 = pd.read_excel(xls, 'Campaign4', index_col ='Date', parse_dates = True)

campaigns = [campaign1,campaign2,campaign3,campaign4]
title_names = ['campaign1', 'campaign2' , 'campaign3', 'campaign4']

#resampling each campaign into weekly
campaign1_weekly = campaign1.resample('W').sum()
campaign2_weekly = campaign2.resample('W').sum()
campaign3_weekly = campaign3.resample('W').sum()
campaign4_weekly = campaign4.resample('W').sum()

#dropping rows that do not have full week of data
campaign1_weekly = campaign1_weekly.drop(campaign1_weekly.index[0])
campaign3_weekly = campaign3_weekly.drop(campaign3_weekly.index[0])
campaign2_weekly = campaign2_weekly.drop(campaign2_weekly.index[0])
""" collected the data on Monday, March 9 so the last week of data is 
complete. Dropping the first week of campaign2 because since the data is too low, 
may throw off the forecast and that week in the other years are not as low """ 

import statsmodels.api as sm
from pylab import rcParams 

#loop for daily decomposition
for a,t in zip(campaigns,title_names):
    rcParams['figure.figsize']=11,9
    decomposition = sm.tsa.seasonal_decompose(a['Queued'])
    fig = decomposition.plot()
    plt.title(t + ' Decomposition Daily')
    plt.show()

#loop for weekly decomposition
weekly_campaigns2 = [campaign1_weekly, campaign2_weekly, campaign3_weekly]
for w,t in zip(weekly_campaigns2,title_names):
    rcParams['figure.figsize']=11,9
    weekly_decomp = sm.tsa.seasonal_decompose(w['Queued'])
    fig = weekly_decomp.plot()
    plt.title(t + ' Decomposition Weekly')
    plt.show()
    #extracting the seasonal graph from the 3 campaigns
    weekly_decomp.seasonal.plot()
    plt.title(t + ' Seasonality')
    plt.show()
    
#resampling each campaign into monthly
campaign1_monthly = campaign1.resample('M').sum()
campaign2_monthly = campaign2.resample('M').sum()
campaign3_monthly = campaign3.resample('M').sum()
campaign4_monthly = campaign4.resample('M').sum()    

""" have to drop the first month for campaign2 and campaign4 as do not have 
full month of data, also have to drop the last row because don't have March completely
so will use February 2020 as the last full month of data """ 

#dropping rows that may skew the data
campaign1_monthly = campaign1_monthly.drop(campaign1_monthly.index[-1])
campaign2_monthly = campaign2_monthly.drop(campaign2_monthly.index[[0,-1]])
campaign3_monthly = campaign3_monthly.drop(campaign3_monthly.index[-1])
campaign4_monthly = campaign4_monthly.drop(campaign4_monthly.index[[0,-1]])

#loop for monthly decomposition
monthly_campaigns = [campaign1_monthly, campaign2_monthly, campaign3_monthly]
for m,t in zip(monthly_campaigns,title_names):
    rcParams['figure.figsize']=11,9
    monthly_decomp = sm.tsa.seasonal_decompose(m['Queued'])
    fig = monthly_decomp.plot()
    plt.title(t + ' Decomposition Monthly')
    plt.show()
    #extracting the seasonal graph from the 3 campaigns
    monthly_decomp.seasonal.plot()
    plt.title(t + ' Seasonality')
    plt.show()
    
###############################################################################
#Reducing the data of campaign3 to starting 2017
###############################################################################
con_new = campaign3['2017-01-01':]

#resampling to weekly and monthly
con_new_weekly = con_new.resample('W').sum()
con_new_monthly = con_new.resample('M').sum()

#dropping the first row of con weekly
con_new_weekly = con_new_weekly.drop(con_new_weekly.index[0])

#dropping the last tow of con monthly 
con_new_monthly = con_new_monthly.drop(con_new_monthly.index[-1])

#plotting new_campaign3 weekly decomposition
rcParams['figure.figsize']=11,9
weekly_decomp = sm.tsa.seasonal_decompose(con_new_weekly['Queued'])
fig = weekly_decomp.plot()
plt.title('campaign3 From 2017 Decomposition Weekly')
plt.show()
    
###############################################################################
#SARIMA MODELING WEEKLY
###############################################################################
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX

for w,t in zip(weekly_campaigns2,title_names):
    weekly_adfuller_results = adfuller(w['Queued'])
    print(f'{t} :', weekly_adfuller_results) 

""" campaign1's p-value is under 0.05 so no need to transform but should transform 
campaign2 and campaign3 channel """   

#first difference and seasonal difference for campaign2 and campaign3_Channel
campaign2_weekly_diff = campaign2_weekly.diff().diff(52).dropna()
campaign3_weekly_diff = campaign3_weekly.diff().diff(52).dropna()

#retesting
campaign2_adfuller = adfuller(campaign2_weekly_diff['Queued'])
print(campaign2_adfuller)

campaign3_adfuller = adfuller(campaign3_weekly_diff['Queued'])
print(campaign3_adfuller) 

"p-value is good with one difference for both campaigns"

#automation
import pmdarima as pm

#auto for campaign1
campaign1_auto = pm.auto_arima(campaign1_weekly, 
                          seasonal= True, m= 52, 
                          d=0, D=1, 
                          max_p = 2, max_q= 2, 
                          trace = True, 
                          error_action = 'ignore',
                          surpress_warnings=True)
print(campaign1_auto.summary())

""" campaign1's best order right now is (1,0,2)(2,1,1,52) which is the same as of 
the March 2020 Forecast and April 2020 Forecast done last week """

#auto for campaign2 and campaign3_Channel
for c1 in [campaign2_weekly,campaign3_weekly]:
    auto_results = pm.auto_arima(c1,
                             seasonal=True, m =52,
                             d=1, D=1,
                             max_p=2,max_q=2,
                             trace = True,
                             error_action = 'ignore',
                             surpress_warnings=True)
    print(auto_results.summary())
    
""" Campaign2's order is (0,1,1)(0,1,0,52) and campaign3's (0,1,0)(2,1,1,52), 
campaign2's changed and campaign3's stayed the same"""

#auto for con_new_weekly
con_auto = pm.auto_arima(con_new_weekly, 
                          seasonal= True, m= 52, 
                          d=1, D=1, 
                          max_p = 2, max_q= 2, 
                          trace = True, 
                          error_action = 'ignore',
                          surpress_warnings=True)
print(con_auto.summary())  
" Con_New_Weekly's order is (1,1,1)(2,1,0,52)"

#instantiating the weekly models
campaign1_weekly_model = SARIMAX(campaign1_weekly, order = (1,0,2), 
                            seasonal_order = (2,1,1,52))
campaign1_weekly_results = campaign1_weekly_model.fit()

campaign2_weekly_model = SARIMAX(campaign2_weekly, order = (0,1,1), 
                           seasonal_order = (0,1,0,52))
campaign2_weekly_results = campaign2_weekly_model.fit()

campaign3_weekly_model = SARIMAX(campaign3_weekly, order = (0,1,0),
                                seasonal_order=(2,1,1,52))
campaign3_weekly_results=campaign3_weekly_model.fit()

con_weekly_model = SARIMAX(con_new_weekly, order = (1,1,1),
                                seasonal_order=(2,1,0,52))
con_weekly_results=con_weekly_model.fit()

#getting the forecast for the three campaigns plus con_new_weekly
campaign1_weekly_forecast = campaign1_weekly_results.get_forecast(steps=45)
campaign1_weekly_mean = campaign1_weekly_forecast.predicted_mean
campaign1_weekly_conf = campaign1_weekly_forecast.conf_int()
campaign1_weekly_dates = campaign1_weekly_mean.index

campaign2_weekly_forecast = campaign2_weekly_results.get_forecast(steps=45)
campaign2_weekly_mean = campaign2_weekly_forecast.predicted_mean
campaign2_weekly_conf = campaign2_weekly_forecast.conf_int()
campaign2_weekly_dates = campaign2_weekly_mean.index

campaign3_weekly_forecast = campaign3_weekly_results.get_forecast(steps=45)
campaign3_weekly_mean=campaign3_weekly_forecast.predicted_mean
campaign3_weekly_conf = campaign3_weekly_forecast.conf_int()
campaign3_weekly_dates = campaign3_weekly_mean.index

con_weekly_forecast = con_weekly_results.get_forecast(steps=45)
con_weekly_mean=con_weekly_forecast.predicted_mean
con_weekly_conf = con_weekly_forecast.conf_int()
con_weekly_dates = con_weekly_mean.index

#plotting each campaign's forecast
weekly_means = [campaign1_weekly_mean, campaign2_weekly_mean,campaign3_weekly_mean, 
                con_weekly_mean]
weekly_conf =  [campaign1_weekly_conf, campaign2_weekly_conf,campaign3_weekly_conf,
                con_weekly_conf]
for w,wm,wc in zip([campaign1_weekly,campaign2_weekly,campaign3_weekly,con_new_weekly],
                   weekly_means, weekly_conf):
    
    plt.figure()

    plt.plot(w.index.values,w.values,label='past')

    plt.plot(wm.index,wm.values,label = 'predicted')

    plt.fill_between(wm.index,wc.iloc[:,0],wc.iloc[:,1], alpha = 0.2)
    plt.legend()
    plt.show()
    
""" con_new_weekly looked better when was testing feb's forecast but now 
campaign3_weekly_mean is the one forecasting more which is what im looking for"""

""" Update: With the newest week, con_new_weekly looks better and is giving
a higher forecast which is what I'm looking for so will use con_new_weekly"""

###############################################################################
#adding day of week to campaign1 data
campaign1_copy = campaign1.copy()
campaign1_copy['week_day'] = campaign1_copy.index.dayofweek
"code works, this gives the number of the weekday, starting from Monday with 0"

campaign1_copy['weekday_name'] = campaign1_copy.index.weekday_name  

#adding day of week to campaign2 data
campaign2_copy = campaign2.copy()
campaign2_copy = campaign2_copy.drop(campaign2_copy.index[0:7])
campaign2_copy['week_day'] = campaign2_copy.index.dayofweek
campaign2_copy['weekday_name'] = campaign2_copy.index.weekday_name  

#adding day of week to campaign3 data
campaign3_copy = campaign3.copy()
campaign3_copy['week_day'] = campaign3_copy.index.dayofweek
campaign3_copy['weekday_name'] = campaign3_copy.index.weekday_name

##############################################################################
#Weekday Percentage Calculation
##############################################################################
     
Mondays_campaign1 = campaign1_copy[campaign1_copy.week_day == 0]
Tuesdays_campaign1 = campaign1_copy[campaign1_copy.week_day == 1]
Wednesdays_campaign1 = campaign1_copy[campaign1_copy.week_day == 2]
Thursdays_campaign1 = campaign1_copy[campaign1_copy.week_day == 3]
Fridays_campaign1 = campaign1_copy[campaign1_copy.week_day == 4]
Saturdays_campaign1 = campaign1_copy[campaign1_copy.week_day == 5]
Sundays_campaign1 = campaign1_copy[campaign1_copy.week_day == 6]
    
#dropping first Friday, Saturday, Sunday
Fridays_campaign1 = Fridays_campaign1.drop(Fridays_campaign1.index[0])
Saturdays_campaign1 = Saturdays_campaign1.drop(Saturdays_campaign1.index[0])
Sundays_campaign1 = Sundays_campaign1.drop(Sundays_campaign1.index[0]) 

""" all days of the week have 218 days which matches perfectly with campaign1_weekly 
that also has 218 observations """   

#getting the values from Mondays to allow division
monday = Mondays_campaign1['Queued'].values
#putting it back into a dataframe
monday =pd.DataFrame(monday,columns=['Queued'])
#division rounded to 2 decimal places
monday_perc_campaign1 = pd.DataFrame(columns=['percentage'])
monday_perc_campaign1['percentage'] = monday['Queued'].div(campaign1_weekly['Queued'].values).mul(100).round(2)
monday_perc_campaign1.mean()

#repeating the process for Tuesday
tuesday = Tuesdays_campaign1['Queued'].values
#putting it back into a dataframe
tuesday =pd.DataFrame(tuesday,columns=['Queued'])
#division rounded to 2 decimal places
tuesday_perc_campaign1 = pd.DataFrame(columns=['percentage'])
tuesday_perc_campaign1['percentage'] = tuesday['Queued'].div(campaign1_weekly['Queued'].values).mul(100).round(2)
tuesday_perc_campaign1.mean()

#repeating the process for Wednesday
wednesday = Wednesdays_campaign1['Queued'].values
#putting it back into a dataframe
wednesday =pd.DataFrame(wednesday,columns=['Queued'])
#division rounded to 2 decimal places
wednesday_perc_campaign1 = pd.DataFrame(columns=['percentage'])
wednesday_perc_campaign1['percentage'] = wednesday['Queued'].div(campaign1_weekly['Queued'].values).mul(100).round(2)
wednesday_perc_campaign1.mean()

#repeating the prcoess for Thursday
thursday = Thursdays_campaign1['Queued'].values
#putting it back into a dataframe
thursday =pd.DataFrame(thursday,columns=['Queued'])
#division rounded to 2 decimal places
thursday_perc_campaign1 = pd.DataFrame(columns=['percentage'])
thursday_perc_campaign1['percentage'] = thursday['Queued'].div(campaign1_weekly['Queued'].values).mul(100).round(2)
thursday_perc_campaign1.mean()

#repeating the prcoess for Friday
friday = Fridays_campaign1['Queued'].values
#putting it back into a dataframe
friday =pd.DataFrame(friday,columns=['Queued'])
#division rounded to 2 decimal places
friday_perc_campaign1 = pd.DataFrame(columns=['percentage'])
friday_perc_campaign1['percentage'] = friday['Queued'].div(campaign1_weekly['Queued'].values).mul(100).round(2)
friday_perc_campaign1.mean()

#repeating the prcoess for Saturday
saturday = Saturdays_campaign1['Queued'].values
#putting it back into a dataframe
saturday = pd.DataFrame(saturday,columns=['Queued'])
#division rounded to 6 decimal places
saturday_perc_campaign1 = pd.DataFrame(columns=['percentage'])
saturday_perc_campaign1['percentage'] = saturday['Queued'].div(campaign1_weekly['Queued'].values).mul(100).round(6)
saturday_perc_campaign1.mean()

#repeating the prcoess for Sunday
sunday = Sundays_campaign1['Queued'].values
#putting it back into a dataframe
sunday = pd.DataFrame(sunday,columns=['Queued'])
#division rounded to 6 decimal places
sunday_perc_campaign1 = pd.DataFrame(columns=['percentage'])
sunday_perc_campaign1['percentage'] = sunday['Queued'].div(campaign1_weekly['Queued'].values).mul(100).round(6)
sunday_perc_campaign1.mean()

#adding all the means together
(monday_perc_campaign1.mean() + tuesday_perc_campaign1.mean() + wednesday_perc_campaign1.mean() + 
 thursday_perc_campaign1.mean() + friday_perc_campaign1.mean() + saturday_perc_campaign1.mean() + 
 sunday_perc_campaign1.mean() )

"percentage is just a tad bit over 100 so seems to be good"

#creating a list to loop over for daily calculation
perc = [monday_perc_campaign1,tuesday_perc_campaign1, wednesday_perc_campaign1, 
        thursday_perc_campaign1, friday_perc_campaign1, saturday_perc_campaign1, 
        sunday_perc_campaign1]

#getting the week of year to see if can help with analysis
campaign1_weekly_copy = campaign1_weekly.copy()
campaign1_weekly_copy['Week_of_Year'] = campaign1_weekly_copy.index.weekofyear   

#printing tail to see what weeks we want to forecast
print(campaign1_weekly_copy.tail())

""" currently have up to week 9 of 2020 so will be forecasting 12 weeks 
starting from week 11 """

weeks = list(range(11,24))

#loop to add week of year column to percs
for p in perc:
    p['week_of_year'] = campaign1_weekly_copy['Week_of_Year'].values

perc_avg = []
#loop to get average day of week percentage for each week to forecast
for w in weeks:
    for p in perc:
        temp = p[p.week_of_year == w]
        perc_avg.append(temp['percentage'].median())

perc_avg[:] = [x / 100 for x in perc_avg]

#creating empty list
campaign1_daily = []
#calculating the daily forecast
start = -7
for x in range(12): #number that is changed based on weeks forecasting
    start = start + 7
    for y in range(7):
        campaign1_daily.append(campaign1_weekly_mean.iloc[x] * perc_avg[y + start])

#converting to dataframe
campaign1_daily = pd.DataFrame(campaign1_daily, columns = ['Forecast'])   
campaign1_daily['date'] = pd.date_range(start='3/9/2020', 
                                      periods=len(campaign1_daily), 
                                      freq='D')  
campaign1_daily = campaign1_daily.set_index('date')
print(campaign1_daily) 

""" might have to beef up the forecast a little because been getting more 
traffic on the end weeks and have been underforecast for the last week of Feb.
Also need to look if Good Friday may impact forecast """

#####repeating the whole process for campaign2 #######

Mondays_campaign2 = campaign2_copy[campaign2_copy.week_day == 0]
Tuesdays_campaign2 = campaign2_copy[campaign2_copy.week_day == 1]
Wednesdays_campaign2 = campaign2_copy[campaign2_copy.week_day == 2]
Thursdays_campaign2 = campaign2_copy[campaign2_copy.week_day == 3]
Fridays_campaign2 = campaign2_copy[campaign2_copy.week_day == 4]
Saturdays_campaign2 = campaign2_copy[campaign2_copy.week_day == 5]
Sundays_campaign2 = campaign2_copy[campaign2_copy.week_day == 6]

""" all days of the week have 128 days which matches perfectly with campaign2_weekly 
that also has 128 observations. Not 129 because dropped the first week of campaign2 """

#getting the values from Mondays to allow division
monday = Mondays_campaign2['Queued'].values
#putting it back into a dataframe
monday =pd.DataFrame(monday,columns=['Queued'])
#division rounded to 2 decimal places
monday_perc_campaign2 = pd.DataFrame(columns=['percentage'])
monday_perc_campaign2['percentage'] = monday['Queued'].div(campaign2_weekly['Queued'].values).mul(100).round(2)
monday_perc_campaign2.mean()

#repeating the process for Tuesday
tuesday = Tuesdays_campaign2['Queued'].values
#putting it back into a dataframe
tuesday =pd.DataFrame(tuesday,columns=['Queued'])
#division rounded to 2 decimal places
tuesday_perc_campaign2 = pd.DataFrame(columns=['percentage'])
tuesday_perc_campaign2['percentage'] = tuesday['Queued'].div(campaign2_weekly['Queued'].values).mul(100).round(2)
tuesday_perc_campaign2.mean()

#repeating the process for Wednesday
wednesday = Wednesdays_campaign2['Queued'].values
#putting it back into a dataframe
wednesday =pd.DataFrame(wednesday,columns=['Queued'])
#division rounded to 2 decimal places
wednesday_perc_campaign2 = pd.DataFrame(columns=['percentage'])
wednesday_perc_campaign2['percentage'] = wednesday['Queued'].div(campaign2_weekly['Queued'].values).mul(100).round(2)
wednesday_perc_campaign2.mean()

#repeating the prcoess for Thursday
thursday = Thursdays_campaign2['Queued'].values
#putting it back into a dataframe
thursday =pd.DataFrame(thursday,columns=['Queued'])
#division rounded to 2 decimal places
thursday_perc_campaign2 = pd.DataFrame(columns=['percentage'])
thursday_perc_campaign2['percentage'] = thursday['Queued'].div(campaign2_weekly['Queued'].values).mul(100).round(2)
thursday_perc_campaign2.mean()

#repeating the prcoess for Friday
friday = Fridays_campaign2['Queued'].values
#putting it back into a dataframe
friday =pd.DataFrame(friday,columns=['Queued'])
#division rounded to 2 decimal places
friday_perc_campaign2 = pd.DataFrame(columns=['percentage'])
friday_perc_campaign2['percentage'] = friday['Queued'].div(campaign2_weekly['Queued'].values).mul(100).round(2)
friday_perc_campaign2.mean()

#repeating the prcoess for Saturday
saturday = Saturdays_campaign2['Queued'].values
#putting it back into a dataframe
saturday = pd.DataFrame(saturday,columns=['Queued'])
#division rounded to 6 decimal places
saturday_perc_campaign2 = pd.DataFrame(columns=['percentage'])
saturday_perc_campaign2['percentage'] = saturday['Queued'].div(campaign2_weekly['Queued'].values).mul(100).round(6)
saturday_perc_campaign2.mean()

#repeating the prcoess for Sunday
sunday = Sundays_campaign2['Queued'].values
#putting it back into a dataframe
sunday = pd.DataFrame(sunday,columns=['Queued'])
#division rounded to 6 decimal places
sunday_perc_campaign2 = pd.DataFrame(columns=['percentage'])
sunday_perc_campaign2['percentage'] = sunday['Queued'].div(campaign2_weekly['Queued'].values).mul(100).round(6)
sunday_perc_campaign2.mean()

#adding all the means together
(monday_perc_campaign2.mean() + tuesday_perc_campaign2.mean() + wednesday_perc_campaign2.mean() + 
 thursday_perc_campaign2.mean() + friday_perc_campaign2.mean() + saturday_perc_campaign2.mean() + 
 sunday_perc_campaign2.mean() )

"percentage is just under 100 so will work for now "

campaign2_weekly_copy = campaign2_weekly.copy()
campaign2_weekly_copy['Week_of_Year'] = campaign2_weekly_copy.index.weekofyear

#printing tail to see what weeks we want to forecast
print(campaign2_weekly_copy.tail())

" going to forecast 12 weeks starting on week 10"

weeks_campaign2 = list(range(11,24))
 
#loop to add week of year column to percs
perc_campaign2 = [monday_perc_campaign2,tuesday_perc_campaign2, wednesday_perc_campaign2, 
        thursday_perc_campaign2, friday_perc_campaign2, saturday_perc_campaign2, 
        sunday_perc_campaign2]

for p in perc_campaign2:
    p['week_of_year'] = campaign2_weekly_copy['Week_of_Year'].values

perc_avg_campaign2 = []
#loop to get average day of week percentage for each week to forecast
for w in weeks_campaign2:
    for p in perc_campaign2:
        temp = p[p.week_of_year == w]
        perc_avg_campaign2.append(temp['percentage'].median())
        
perc_avg_campaign2[:] = [x / 100 for x in perc_avg_campaign2]      

campaign2_daily = []
#calculating the daily forecast
start = -7
for x in range(12):
    start = start + 7
    for y in range(7):
        campaign2_daily.append(campaign2_weekly_mean.iloc[x] * perc_avg_campaign2[y + start]) 
        
   
        
#converting campaign2_daily to a dataframe to add date indexes to forecast
campaign2_daily = pd.DataFrame(campaign2_daily, columns = ['Forecast'])   
campaign2_daily['date'] = pd.date_range(start='3/9/2020', 
                                      periods=len(campaign2_daily), 
                                      freq='D')  
campaign2_daily = campaign2_daily.set_index('date')
print(campaign2_daily)

""" will take a closer look at the numbers in excel, will see how Good Friday 
may impact forecast """

######## repeating the process for campaign3 channel ############

Mondays_campaign3 = campaign3_copy[campaign3_copy.week_day == 0]
Tuesdays_campaign3 = campaign3_copy[campaign3_copy.week_day == 1]
Wednesdays_campaign3 = campaign3_copy[campaign3_copy.week_day == 2]
Thursdays_campaign3 = campaign3_copy[campaign3_copy.week_day == 3]
Fridays_campaign3 = campaign3_copy[campaign3_copy.week_day == 4]
Saturdays_campaign3 = campaign3_copy[campaign3_copy.week_day == 5]
Sundays_campaign3 = campaign3_copy[campaign3_copy.week_day == 6]

#dropping first Friday, Saturday, Sunday
Fridays_campaign3 = Fridays_campaign3.drop(Fridays_campaign3.index[0])
Saturdays_campaign3 = Saturdays_campaign3.drop(Saturdays_campaign3.index[0])
Sundays_campaign3 = Sundays_campaign3.drop(Sundays_campaign3.index[0])

""" all days of the week have 218 days which matches perfectly with 
campaign3_weekly that also has 218 observations """

#getting the values from Mondays to allow division
monday = Mondays_campaign3['Queued'].values
#putting it back into a dataframe
monday =pd.DataFrame(monday,columns=['Queued'])
#division rounded to 2 decimal places
monday_perc_campaign3 = pd.DataFrame(columns=['percentage'])
monday_perc_campaign3['percentage'] = monday['Queued'].div(campaign3_weekly['Queued'].values).mul(100).round(2)
monday_perc_campaign3.mean()

#repeating the process for Tuesday
tuesday = Tuesdays_campaign3['Queued'].values
#putting it back into a dataframe
tuesday =pd.DataFrame(tuesday,columns=['Queued'])
#division rounded to 2 decimal places
tuesday_perc_campaign3 = pd.DataFrame(columns=['percentage'])
tuesday_perc_campaign3['percentage'] = tuesday['Queued'].div(campaign3_weekly['Queued'].values).mul(100).round(2)
tuesday_perc_campaign3.mean()

#repeating the process for Wednesday
wednesday = Wednesdays_campaign3['Queued'].values
#putting it back into a dataframe
wednesday =pd.DataFrame(wednesday,columns=['Queued'])
#division rounded to 2 decimal places
wednesday_perc_campaign3 = pd.DataFrame(columns=['percentage'])
wednesday_perc_campaign3['percentage'] = wednesday['Queued'].div(campaign3_weekly['Queued'].values).mul(100).round(2)
wednesday_perc_campaign3.mean()

#repeating the prcoess for Thursday
thursday = Thursdays_campaign3['Queued'].values
#putting it back into a dataframe
thursday =pd.DataFrame(thursday,columns=['Queued'])
#division rounded to 2 decimal places
thursday_perc_campaign3 = pd.DataFrame(columns=['percentage'])
thursday_perc_campaign3['percentage'] = thursday['Queued'].div(campaign3_weekly['Queued'].values).mul(100).round(2)
thursday_perc_campaign3.mean()

#repeating the prcoess for Friday
friday = Fridays_campaign3['Queued'].values
#putting it back into a dataframe
friday =pd.DataFrame(friday,columns=['Queued'])
#division rounded to 2 decimal places
friday_perc_campaign3 = pd.DataFrame(columns=['percentage'])
friday_perc_campaign3['percentage'] = friday['Queued'].div(campaign3_weekly['Queued'].values).mul(100).round(2)
friday_perc_campaign3.mean()

#repeating the prcoess for Saturday
saturday = Saturdays_campaign3['Queued'].values
#putting it back into a dataframe
saturday = pd.DataFrame(saturday,columns=['Queued'])
#division rounded to 6 decimal places
saturday_perc_campaign3 = pd.DataFrame(columns=['percentage'])
saturday_perc_campaign3['percentage'] = saturday['Queued'].div(campaign3_weekly['Queued'].values).mul(100).round(6)
saturday_perc_campaign3.mean()

#repeating the prcoess for Sunday
sunday = Sundays_campaign3['Queued'].values
#putting it back into a dataframe
sunday = pd.DataFrame(sunday,columns=['Queued'])
#division rounded to 6 decimal places
sunday_perc_campaign3 = pd.DataFrame(columns=['percentage'])
sunday_perc_campaign3['percentage'] = sunday['Queued'].div(campaign3_weekly['Queued'].values).mul(100).round(6)
sunday_perc_campaign3.mean()

#adding all the means together
(monday_perc_campaign3.mean() + tuesday_perc_campaign3.mean() + 
 wednesday_perc_campaign3.mean() + thursday_perc_campaign3.mean() + 
 friday_perc_campaign3.mean() + saturday_perc_campaign3.mean() + 
 sunday_perc_campaign3.mean() )

" get just a little over 100 so is good for now "

campaign3_weekly_copy = campaign3_weekly.copy()
campaign3_weekly_copy['Week_of_Year'] = campaign3_weekly_copy.index.weekofyear

#printing tail to see what weeks we want to forecast
print(campaign3_weekly_copy.tail())

" going to forecast 12 weeks starting on week 10 "

weeks_campaign3 = list(range(11,24))
 
#loop to add week of year column to percs
perc_campaign3 = [monday_perc_campaign3,tuesday_perc_campaign3, 
                 wednesday_perc_campaign3, thursday_perc_campaign3, 
                 friday_perc_campaign3, saturday_perc_campaign3, 
                 sunday_perc_campaign3]

for p in perc_campaign3:
    p['week_of_year'] = campaign3_weekly_copy['Week_of_Year'].values

perc_avg_campaign3 = []
#loop to get average day of week percentage for each week to forecast
for w in weeks_campaign3:
    for p in perc_campaign3:
        temp = p[p.week_of_year == w]
        perc_avg_campaign3.append(temp['percentage'].median())
        
perc_avg_campaign3[:] = [x / 100 for x in perc_avg_campaign3]      

#creating an empty list
campaign3_daily = []
#calculating the daily forecast
start = -7
for x in range(12):
    start = start + 7
    for y in range(7):
        campaign3_daily.append(con_weekly_mean.iloc[x] * perc_avg_campaign3[y + start])
        #line above is where changed campaign3_weekly_mean to con_weekly_mean        
        
#converting campaign3_daily to a dataframe to add date indexes to forecast
campaign3_daily = pd.DataFrame(campaign3_daily, columns = ['Forecast'])   
campaign3_daily['date'] = pd.date_range(start='3/9/2020', 
                                      periods=len(campaign3_daily), 
                                      freq='D')  
campaign3_daily = campaign3_daily.set_index('date')
print(campaign3_daily)

#exporting the daily forecasts to an excel
with pd.ExcelWriter('April2020_Forecast_Python_Ver2.xlsx') as writer:
    campaign1_daily.to_excel(writer,sheet_name ='campaign1')
    campaign2_daily.to_excel(writer,sheet_name = 'campaign2')
    campaign3_daily.to_excel(writer,sheet_name='campaign3 Channel')

""" exporting the data but will continue the script to do the 60, 90 day 
forecast with SARIMA monhly and then add manually in Excel """    

###############################################################################
#SARIMA MODELING MONTHLY
###############################################################################

for m,t in zip(monthly_campaigns,title_names):
    monthly_adfuller_results = adfuller(m['Queued'])
    print(f'{t} :', monthly_adfuller_results)  
    
campaign1_monthly_diff = campaign1_monthly.diff().diff(12).dropna()
campaign2_monthly_diff = campaign2_monthly.diff().diff(12).dropna()
campaign3_monthly_diff = campaign3_monthly.diff().diff(12).dropna()


#retesting
campaign1_adfuller_monthly = adfuller(campaign1_monthly_diff['Queued'])
print(campaign1_adfuller_monthly)

campaign2_adfuller_monthly = adfuller(campaign2_monthly_diff['Queued'])
print(campaign2_adfuller_monthly)

campaign3_adfuller_monthly = adfuller(campaign3_monthly_diff['Queued'])
print(campaign3_adfuller_monthly)   

""" the differncing does not work on campaign2 but the other two it does, will 
proceed as normal but might check with R studio to see what makes most sense"""

#campaign1 auto arima
campaign1_auto_monthly = pm.auto_arima(campaign1_monthly, 
                          seasonal= True, m= 12, 
                          d=1, D=1, 
                          max_p = 2, max_q= 2, 
                          trace = True, 
                          error_action = 'ignore',
                          surpress_warnings=True)
print(campaign1_auto_monthly.summary())     

"campaign1's monthly order is (2,1,1)(1,1,1,12), changed from last time"

#campaign2 auto arima
campaign2_auto_monthly = pm.auto_arima(campaign2_monthly, 
                          seasonal= True, m= 12, 
                          d=1, D=1, 
                          max_p = 2, max_q= 2, 
                          trace = True, 
                          error_action = 'ignore',
                          surpress_warnings=True)
print(campaign2_auto_monthly.summary())

""" campaign2's monthly order is (0,1,0)(0,1,0,12), same as last time. This is most
likely due to not passing the dicky-fuller test"""   

campaign3_auto_monthly = pm.auto_arima(campaign3_monthly, 
                          seasonal= True, m= 12, 
                          d=1, D=1, 
                          max_p = 2, max_q= 2, 
                          trace = True, 
                          error_action = 'ignore',
                          surpress_warnings=True)
print(campaign3_auto_monthly.summary())    

"campaign3's monthly order is (0,1,1)(0,1,1,12), different than last time"


#instantiating the monthly models
campaign1_monthly_model = SARIMAX(campaign1_monthly, order = (2,1,1), 
                            seasonal_order = (1,1,1,12))
campaign1_monthly_results = campaign1_monthly_model.fit()

campaign2_monthly_model = SARIMAX(campaign2_monthly, order = (0,1,0), 
                           seasonal_order = (0,1,0,12))
campaign2_monthly_results = campaign2_monthly_model.fit()

campaign3_monthly_model = SARIMAX(campaign3_monthly, order = (0,1,1),
                                seasonal_order=(0,1,1,12))
campaign3_monthly_results=campaign3_monthly_model.fit()

#getting the forecast for the three campaigns
campaign1_monthly_forecast = campaign1_monthly_results.get_forecast(steps=12)
campaign1_monthly_mean = campaign1_monthly_forecast.predicted_mean
campaign1_monthly_conf = campaign1_monthly_forecast.conf_int()
campaign1_monthly_dates = campaign1_monthly_mean.index

campaign2_monthly_forecast = campaign2_monthly_results.get_forecast(steps=12)
campaign2_monthly_mean = campaign2_monthly_forecast.predicted_mean
campaign2_monthly_conf = campaign2_monthly_forecast.conf_int()
campaign2_monthly_dates = campaign2_monthly_mean.index

campaign3_monthly_forecast = campaign3_monthly_results.get_forecast(steps=12)
campaign3_monthly_mean=campaign3_monthly_forecast.predicted_mean
campaign3_monthly_conf = campaign3_monthly_forecast.conf_int()
campaign3_monthly_dates = campaign3_monthly_mean.index

#plotting each campaign's forecast
monthly_means = [campaign1_monthly_mean, campaign2_monthly_mean,campaign3_monthly_mean]
monthly_conf =  [campaign1_monthly_conf, campaign2_monthly_conf,campaign3_monthly_conf]
for m,mm,mc in zip([campaign1_monthly,campaign2_monthly,campaign3_monthly],monthly_means, monthly_conf):
    
    plt.figure()
    
    plt.ylim(0,100000)
    plt.plot(m.index.values,m.values,label='past')

    plt.plot(mm.index,mm.values,label = 'predicted')

    plt.fill_between(mm.index,mc.iloc[:,0],mc.iloc[:,1], alpha = 0.2)
    plt.legend()
    plt.show()
    
""" have been monitoring traffic and for campaign1, seems like "something" was fixed
and traffic may be more similar to 2018. will try ofrecasting with data up to
Jan 6th 2019 """

###############################################################################
#Reducing the data of Cocampaign1 to until Jan 6 2019
############################################################################### 
campaign1_new = campaign3[:'2019-01-06']

#resampling to weekly and monthly
campaign1_new_weekly = campaign1_new.resample('W').sum()
campaign1_new_monthly = campaign1_new.resample('M').sum()

#dropping the first row of con weekly
campaign1_new_weekly = campaign1_new_weekly.drop(campaign1_new_weekly.index[0])

#dropping the last tow of con monthly 
campaign1_new_monthly = campaign1_new_monthly.drop(campaign1_new_monthly.index[-1])

#plotting new_campaign3 weekly decomposition
rcParams['figure.figsize']=11,9
weekly_decomp = sm.tsa.seasonal_decompose(campaign1_new_weekly['Queued'])
fig = weekly_decomp.plot()
plt.title('campaign1 Until Jan 6 2019 Decomposition Weekly')
plt.show()

###############################################################################
#SARIMA MODELING WEEKLY FOR campaign1_NEW
###############################################################################

#adfuller test
campaign1_new_adfuller = adfuller(campaign1_new_weekly['Queued']) 
print(campaign1_new_adfuller)  
"p-value is just above 0.05, will try differencing to see if helps"

#first difference and seasonal difference for campaign1_New
campaign1_new_diff = campaign1_new_weekly.diff().diff(52).dropna()


#retesting
campaign1_new_adfuller2 = adfuller(campaign1_new_diff['Queued'])
print(campaign1_new_adfuller2)
"one difference does the job"

#auto for campaign1_new
campaign1_new_auto = pm.auto_arima(campaign1_new_weekly, 
                          seasonal= True, m= 52, 
                          d=1, D=1, 
                          max_p = 2, max_q= 2, 
                          trace = True, 
                          error_action = 'ignore',
                          surpress_warnings=True)
print(campaign1_new_auto.summary())
"campaign1_new_weekly's order is (0,1,0)(1,1,0,52)"

#instantiating the model
campaign1_new_weekly_model = SARIMAX(campaign1_new_weekly, order = (0,1,0), 
                            seasonal_order = (1,1,0,52))
campaign1_new_weekly_results = campaign1_new_weekly_model.fit()

#getting the forecast
campaign1_new_weekly_forecast = campaign1_new_weekly_results.get_forecast(steps=52)
campaign1_new_weekly_mean = campaign1_new_weekly_forecast.predicted_mean
campaign1_new_weekly_conf = campaign1_new_weekly_forecast.conf_int()
campaign1_new_weekly_dates = campaign1_new_weekly_mean.index

#plotting the forecast
plt.figure()

plt.plot(campaign1_new_weekly.index.values,campaign1_new_weekly.values,label='past')

plt.plot(campaign1_new_weekly_mean.index,campaign1_new_weekly_mean.values,
         label = 'predicted')

plt.fill_between(campaign1_new_weekly_mean.index,campaign1_new_weekly_conf.iloc[:,0],
                 campaign1_new_weekly_conf.iloc[:,1], alpha = 0.2)
plt.legend()
plt.show()