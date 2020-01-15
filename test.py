# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:07:42 2019

@author: Kaush
"""

################ Twitter Practice #############################


import pandas as pd, numpy as np
import twitter
import datetime
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighbourClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, prediction_score



twitter_key = {
        
'consumer_key': 'L4sziHBqV4VUIfKezbos0JMVl',
'consumer_secret': 'lJau6R7GIHFwoGR5wB3PlLQPXBChwzJFJ9WGXXtazcDSA1Vb1X',
'accesss_token_key': '941359629606539264-05XcmQfdwMXTbPNWS3r7cZThvbQBxCK',
'access_token_secret': 'VdE3VJVk6oxbohQGcw7WYA5Tg4Sr8kW9duTO1wxmB6qXk'
        } 

api = twitter.api(
        consumer_key = twitter_key["consumer_Key"],
        consumer_secret = twitter_key["consumer_secret"],
        access_token_key = twitter_key["access_token_key"],
        access_token_secret = twitter_key["access_token_secret"],
#        twitter_mode = 'extended'
        )

print(type(api))


class twitter_miner(object):

    def __init__(self, api, result_limit =20):
        
        self.api = api
        self.result_limit = result_limit
        
    def mine_user_tweet(self, user = "HilaryClinton", mine_retweet=False, max_pages=20):
        
        data = []
        last_tweet_id = False
        page = 1
        
        while page<= max_pages:
            
            if last_tweet_id:
                statuses = self.api.GetUserTimeline(self, screen_name = user, count = self.result_limit, max_id = last_tweet_id-1, include_rts = mine_retweet)
                statuses = [_.AsDict() for _ in statuses]
            else:
                statuses = self.api.GetUsetTimeline(self, screen_name=user, count = self.result_limit, include_rts=mine_retweet)
                statuses = [_.AsDict() for _ in statuses]
            
            for tweet in statuses:
                
                try:
                    mined = {
                            'tweet_id' : tweet['tweet_id'],
                            'handle' : tweet['user']['screen_name'],
                            'retweet_count' : tweet['retweet_count'],
                            'text' : tweet['full_test'],
                            'mined_at' : datetime.datetime.now(),
                            'created_at' : tweet['created_at']
                            
                            }
                    
                except:
                    mined = {
                            
                            'tweet_id' : tweet['tweet_id'],
                            'handle' : tweet['user']['sreen_name'],
                            'retweet_count' : 0,
                            'text' : tweet['full_text'],
                            'mined_at' : datetime.datetime.now(),
                            'created_at' : tweet['created_at']
                            
                                }
        
                last_tweet_id = tweet['id']
                data.append(mined)
            
            page += 1
            
        return(data)
        
        
        
        
miner = twitter_miner(api, result_limit = 200)


hillary = miner.mine_user_tweet(user='HillaryClinton')
donald = miner.mine_user_tweet(user='realDonaldTrump')



