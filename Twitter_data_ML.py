# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 10:22:00 2019

@author: Kaush
"""

import numpy as np
import twitter, datetime, pandas as pd

twitter_keys = {
'consumer_key': 'L4sziHBqV4VUIfKezbos0JMVl',
'consumer_secret': 'lJau6R7GIHFwoGR5wB3PlLQPXBChwzJFJ9WGXXtazcDSA1Vb1X',
'accesss_token_key': '941359629606539264-05XcmQfdwMXTbPNWS3r7cZThvbQBxCK',
'access_token_secret': 'VdE3VJVk6oxbohQGcw7WYA5Tg4Sr8kW9duTO1wxmB6qXk'
        } 

api = twitter.Twitter(
      auth=OAuth(
        consumer_key = twitter_keys['consumer_key'],
        consumer_secret = twitter_keys['consumer_secret'],
        accesss_token_key = twitter_keys['accesss_token_key'],
        access_token_secret = twitter_keys['access_token_secret'],
        twitter_mode = 'extended'
        ))

print(type(api))
    
    
################################## Class ################################################
    
    
#TweetMiner function from Mike Roman

class TweetMiner(object):

    
    def __init__(self, api, result_limit = 20):
        
        self.api = api        
        self.result_limit = result_limit
        

    def mine_user_tweets(self, user="HillaryClinton", mine_retweets=False, max_pages=20):

        data           =  []
        last_tweet_id  =  False
        page           =  1
        
        while page <= max_pages:
            
            if last_tweet_id:
                statuses   =   self.api.GetUserTimeline(screen_name=user, count=self.result_limit, max_id=last_tweet_id - 1, include_rts=mine_retweets)
                statuses = [ _.AsDict() for _ in statuses]
            else:
                statuses   =   self.api.GetUserTimeline(screen_name=user, count=self.result_limit, include_rts=mine_retweets)
                statuses = [_.AsDict() for _ in statuses]
                
            for item in statuses:
                # Using try except here.
                # When retweets = 0 we get an error (GetUserTimeline fails to create a key, 'retweet_count')
                try:
                    mined = {
                        'tweet_id':        item['id'],
                        'handle':          item['user']['screen_name'],
                        'retweet_count':   item['retweet_count'],
                        'text':            item['full_text'],
                        'mined_at':        datetime.datetime.now(),
                        'created_at':      item['created_at'],
                    }
                
                except:
                        mined = {
                        'tweet_id':        item['id'],
                        'handle':          item['user']['screen_name'],
                        'retweet_count':   0,
                        'text':            item['full_text'],
                        'mined_at':        datetime.datetime.now(),
                        'created_at':      item['created_at'],
                    }
                
                last_tweet_id = item['id']
                data.append(mined)
                
            page += 1
            
        return data
    
    
    

# Result limit == count parameter from out GetUserTimeline()
        
miner = TweetMiner(api, result_limit=200)

hillary = miner.mine_user_tweets(user = 'HillaryClinton')
donald = miner.mine_user_tweets(user = 'realDonaldTrump')


for x in range(5):
    print (hillary[x]['text'])
    print('____----____')
    
    
for x in range(5):
    print( donald[x]['text'])
    print('_____----____')
    
    
###################################  Convert tweet outputs to a pandas DataFrame #######################################
    
####################  Creating training data #####
    
########### Mine data from the twitter api ###
    
### 1. Mine Trump tweets,   2. Create a tweet DataFrame,  3. Mine Hillary tweets,   4. Append the results to our DataFrame.  #####


miner = TweetMiner(api, result_limit=200)
trump_tweets = miner.mine_user_tweets("realDonaldTrump", max_pages = 14)
donald_df = pd.DataFrame(trump_tweets)

print("Donald_df = ", donald_df, "\n ############################################\n", donald_df.shape)
      
      
hillary_tweets = miner.mine_user_tweets("HillaryClinton")
hillary_df = pd.DataFrame(hillary_tweets)
print("Hillary_df = ", hillary_df, '\n ##########################################\n', hillary_df.shape)
      
      
tweets = pd.concat([donald_df, hillary_df], axis = 0)

print("tweet_shape = ", tweets.shape)


############################################ Any interesting ngrams going on with Trump or Hillary? ##########################

from sklearn.feature_extraction.text import TfidVectorizer
from collections import Counter

# We can use the TfidfVectorizer to find ngrams for us
vect = TfidVectorizer(ngram_range=(2, 5), stop_words='english')

# Pulls all of trumps tweet text's into one giant string
summaries = ''.join(donald_df['text'])
ngrams_summaries = vect.build_analyzer()(summaries)

print("Common ngrams = ", Counter(ngrams_summaries).most_common(20) )




############################ Fake news....figures ############################3

vect = TfidVectorizer(ngram_range=(2, 5), stop_words= 'english')

summaries = ''.join(hillary_df['text'])
ngrams_summaries = vect.build_analyzer()(summaries)

print("Common ngrams for hillary = ", Counter(ngrams_summaries).most_common(20))


############################## Processing the tweets and building a model ################################


################ Cleaning the text using textacy ##################

from textacy.preprocess import preprocess_text

tweet_text = tweets['text'].values
clean_text = [preprocess_text(x, fix_unicode=True, lowercase=True, no_urls=True, no_emails=True, no_phone_numbers=True, no_currancy_symbol=True, no_punct=True, no_accents=True) for x in tweet_text]


print("tweet_text = ", tweet_text[1:8])

print("clean_text = ", clean_text[1:8])


################## Creating Target ########################

y = tweets['handle'].map(lambda x: 1 if x == 'realDonaldTrue' else 0). values
print(max(pd.Series(y)).value_counts(normalize=True))



##################### Building Algo ####################

from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.tree import DecionTreeClassifier
#from sklearn.neighbour import KNeighboursClassifier


#Vectorizing with TF-IDF Vectorizer and creating X matrix

tfv = TfidVectorizer(ngram_range=(2,4), max_features=2000)
X = tfv.fit_transform(clean_text).todense()

print("X..shape = ", X.shape)

from sklearn.model_selection import GridSearchCV

lr= LogisticRegression()
params = {'penalty' : ['l1', 'l2'], 'C' : np.logspace(-5, 0, 100)}

#Grid searching to find optimal parameters for Logistic Regression

gs = GridSearchCV(lr, param_grid=params, cv=10, verbose=1)
gs.fit(X, y)

print("gs.best_params_ = ", gs.best_params_)
print("gs.best_score_ = ", gs.best_score_)


from sklearn.model_selection import cross_val_score


accuracies = cross_val_score(LogisticRegression(), X, y, cv=10)

print (accuracies.mean())
print (1-y.mean())


##################################### Check the predicted probability for a random Sanders and Trump tweet #######################


estimator = LogisticRegression(penalty='l2',C=1.0)
estimator.fit(X,y)

# Prep our source as TfIdf vectors
source_test = [
    "The presidency doesn’t change who you are—it reveals who you are. And we’ve seen all we need to of Donald Trump.",
    "Crooked Hillary is spending tremendous amounts of Wall Street money on false ads against me. She is a very dishonest person!"
]

###
# NOTE:  Do not re-initialize the tfidf vectorizor or the feature space will be overwritten and
# your transform will not match the number of features you trained your model on.
#
# This is why you only need to "transform" since you already "fit" previously
#
####

Xtest = tfv.transform(source_test)
pd.DataFrame(estimator.predict_proba(Xtest), columns=["Proba_Hillary", "Proba_Trump"])


############################################################################# 
#Now I'm going to to attempt to extract the tweets that have the highest and lowest probability of being from Trump or Hillary based on the model.
#I'm going to do this by:
#Using the Predict Proba method to give me an array of the probabilites of Hillary and Trump tweets
#Transform that array into a dataframe
#Merge the tweets datafram and probability dataframe
#Filter and create dataframe with only tweets of either person
#Use a list comprehension to print out the highest and lowest probability tweets


print(estimator.predict_proba(X))

Probas_x = pd.DataFrame(estimator.predict_proba(X), columns=["Proba_Hillary", "Proba_Donald"])

joined_x = pd.merge(tweets, Probas_x, left_index=True, right_index=True)

print(joined_x)

joined_hillary = joined_x[joined_x['handle']=="HillaryClinton"]
for el in joined_hillary[joined_hillary['Proba_Hillary']==max(joined_hillary['Proba_Hillary'])]['text']:
    print (el)
    

for el in joined_hillary[joined_hillary['Proba_Hillary']==min(joined_hillary['Proba_Hillary'])]['text']:
    print (el)    


joined_donald = joined_x[joined_x['handle']=="realDonaldTrump"]
for el in joined_donald[joined_donald['Proba_Donald']==max(joined_donald['Proba_Donald'])]['text']:
    print (el)

for el in joined_donald[joined_donald['Proba_Donald']==min(joined_donald['Proba_Donald'])]['text']:
    print (el)



