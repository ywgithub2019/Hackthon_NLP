# -*- coding: utf-8 -*-


# install the package
# https://pypi.org/project/pytrends/

pip install pytrends

from pytrends.request import TrendReq

## Login to Google. Only need to run this once, the rest of requests will use the same session.
# tz:Timezone Offset. For example US CST is `'360'`

pytrends = TrendReq(hl='en-US', tz=360)

## Create payload and capture API tokens. Only needed for interest_over_time(), interest_by_region() & related_queries()
pytrends.build_payload(kw_list=['bitcoin'],timeframe='2010-05-01 2018-04-30')
# In order to get the monthly data from 2017-05-01 to 2018-05001, I decide to extend the time period and select the time period needed 
 

## Interest Over Time
# returns historical, indexed data for when the keyword was searched most as shown on Google Trends’ Interest Over Time section.
bitcoin_2017_2018 = pytrends.interest_over_time()
bitcoin_2017_2018 = bitcoin_2017_2018.drop(columns = 'isPartial')
bitcoin_2017_2018.head()
print(bitcoin_2017_2018)

"""### extract data by month to get the daily data"""

# '2017-05-01 2017-05-31'

pytrends_201705 = TrendReq(hl='en-US', tz=360)
pytrends_201705.build_payload(kw_list=['bitcoin'],timeframe='2017-05-01 2017-05-31')
bitcoin_201705 =pytrends_201705.interest_over_time()
bitcoin_201705 = bitcoin_201705.drop(columns = 'isPartial')
print(bitcoin_201705)

# '2017-06-01 2017-06-30'

pytrends_201706 = TrendReq(hl='en-US', tz=360)
pytrends_201706.build_payload(kw_list=['bitcoin'],timeframe='2017-06-01 2017-06-30')
bitcoin_201706 =pytrends_201706.interest_over_time()
bitcoin_201706 = bitcoin_201706.drop(columns = 'isPartial')
print(bitcoin_201706)

# '2017-07-01 2017-07-31'

pytrends_201707 = TrendReq(hl='en-US', tz=360)
pytrends_201707.build_payload(kw_list=['bitcoin'],timeframe='2017-07-01 2017-07-31')
bitcoin_201707 =pytrends_201707.interest_over_time()
bitcoin_201707 = bitcoin_201707.drop(columns = 'isPartial')
print(bitcoin_201707)

# '2017-08-01 2017-08-31'

pytrends_201708 = TrendReq(hl='en-US', tz=360)
pytrends_201708.build_payload(kw_list=['bitcoin'],timeframe='2017-08-01 2017-08-31')
bitcoin_201708 =pytrends_201708.interest_over_time()
bitcoin_201708 = bitcoin_201708.drop(columns = 'isPartial')
print(bitcoin_201708)

# '2017-09-01 2017-09-30'

pytrends_201709 = TrendReq(hl='en-US', tz=360)
pytrends_201709.build_payload(kw_list=['bitcoin'],timeframe='2017-09-01 2017-09-30')
bitcoin_201709 =pytrends_201709.interest_over_time()
bitcoin_201709 = bitcoin_201709.drop(columns = 'isPartial')
print(bitcoin_201709)

# '2017-10-01 2017-10-31'

pytrends_201710 = TrendReq(hl='en-US', tz=360)
pytrends_201710.build_payload(kw_list=['bitcoin'],timeframe='2017-10-01 2017-10-31')
bitcoin_201710 =pytrends_201710.interest_over_time()
bitcoin_201710 = bitcoin_201710.drop(columns = 'isPartial')
print(bitcoin_201710)

# '2017-11-01 2017-11-30'

pytrends_201711 = TrendReq(hl='en-US', tz=360)
pytrends_201711.build_payload(kw_list=['bitcoin'],timeframe='2017-11-01 2017-11-30')
bitcoin_201711 =pytrends_201711.interest_over_time()
bitcoin_201711 = bitcoin_201711.drop(columns = 'isPartial')
print(bitcoin_201711)

# '2017-12-01 2017-12-31'

pytrends_201712 = TrendReq(hl='en-US', tz=360)
pytrends_201712.build_payload(kw_list=['bitcoin'],timeframe='2017-12-01 2017-12-31')
bitcoin_201712 =pytrends_201712.interest_over_time()
bitcoin_201712 = bitcoin_201712.drop(columns = 'isPartial')
print(bitcoin_201712)

# '2018-01-01 2018-01-31'

pytrends_201801 = TrendReq(hl='en-US', tz=360)
pytrends_201801.build_payload(kw_list=['bitcoin'],timeframe='2018-01-01 2018-01-31')
bitcoin_201801 =pytrends_201801.interest_over_time()
bitcoin_201801 = bitcoin_201801.drop(columns = 'isPartial')
print(bitcoin_201801)

# '2018-02-01 2018-02-28'

pytrends_201802 = TrendReq(hl='en-US', tz=360)
pytrends_201802.build_payload(kw_list=['bitcoin'],timeframe='2018-02-01 2018-02-28')
bitcoin_201802 =pytrends_201802.interest_over_time()
bitcoin_201802 = bitcoin_201802.drop(columns = 'isPartial')
print(bitcoin_201802)

# '2018-03-01 2018-03-31'

pytrends_201803 = TrendReq(hl='en-US', tz=360)
pytrends_201803.build_payload(kw_list=['bitcoin'],timeframe='2018-03-01 2018-03-31')
bitcoin_201803 =pytrends_201803.interest_over_time()
bitcoin_201803 = bitcoin_201803.drop(columns = 'isPartial')
print(bitcoin_201803)

# '2018-04-01 2018-04-30'

pytrends_201804 = TrendReq(hl='en-US', tz=360)
pytrends_201804.build_payload(kw_list=['bitcoin'],timeframe='2018-04-01 2018-04-30')
bitcoin_201804 =pytrends_201804.interest_over_time()
bitcoin_201804 = bitcoin_201804.drop(columns = 'isPartial')
print(bitcoin_201804)

import pandas as pd
frames = [bitcoin_201705,bitcoin_201706,bitcoin_201707,bitcoin_201708,bitcoin_201709,bitcoin_201710,bitcoin_201711,bitcoin_201712,bitcoin_201801,bitcoin_201802,bitcoin_201803,bitcoin_201804]
google_daily = pd.concat(frames)
len(google_daily)

google_daily.to_excel("google_daily.xlsx")

bitcoin_2017_2018.to_excel("google_monthly.xlsx")



