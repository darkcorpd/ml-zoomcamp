# Midterm project
for the [Machine Learning Zoomcamp](https://mlzoomcamp.com) by Alexey Grigorev.

## Problem description

### Twitter bot detection using supervised machine learning
In the world of Internet and social media, there are about 3.8 billion active social media users and 4.5 billion people accessing the internet daily. Every year there is a 9% growth in the number of users and half of the internet traffic consists of mostly bots.

Malicious bots make up 20% of the traffic, they are used for nefarious purposes, they can mimic human behavior, they can impersonate legal traffic, attack IoT devices and exploit their performance. 
Among all these concerns, the primary concern is for social media users as they represent a large group of active users on the internet, they are more vulnerable to breach of data, change in opinion based on data. Detection of such bots is crucial to prevent further mishaps. 

I use supervised Machine learning techniques in this project such as Logistic regression, Decision tree, Random Forest and XGBoost to calculate their accuracies and compare to detect Twitter bots from a given training data set. 

## Data collection

I have found a [dataset](https://github.com/antibot4navalny/accounts_labelled/blob/main/labels.json) of accounts recognized as Twitter bots. The dataset is used in Chrome extention '[MetaBot for Twitter](https://github.com/antibot4navalny/metabot)' to highlight bots on Twitter.

To get a list of account names from the JSON I simply subset even keys.

After that I used a `twitter_scraper_selenium` library to get profile details.

As I have collected some data before I use list comparation to update information only for new names in bot list (updatable).

My recent bots dataset contains 773 rows × 122 columns.

The same way is used to collected data on real organic users provided by the same author `antibot4navalny`. Two datasets of bot and organic users' profiles are mixed to get a train set for a bot classification problem.

## Exploratory data analysis (EDA) 
is an especially important activity in the routine of a data analyst or scientist. It enables an in depth understanding of the dataset, define or discard hypotheses and create predictive models on a solid basis.

First of all I remove duplicates.

Then as some of the columns of dataframe are actually empty we can drop them too. To find a thereshold by which I will drop the columns I've plotted a distribution chart for NaN values. All the columns that have more than 2 NaN values can be removed.

After that I have explored the data by the common approach:
*	**Variable:** name of the variable.
*	**Type:** the type or format of the variable. This can be categorical, numeric, Boolean, and so on.
*	**Context:** useful information to understand the semantic space of the variable. In the case of our dataset, the context is the social one.
*	**Expectation:** how relevant is this variable with respect to our task? We can use a scale “High, Medium, Low”.
*	**Comments:** whether or not we have any comments to make on the variable.
https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/user
---  ------                             --------------  -----  
0. **id** - int64 - 585 - The integer representation of the unique identifier for this User - **index**
1. **id_str** - int64 - 585 - The string representation of the unique identifier for this User. - **drop**
2. **name** - object - 582 - The name of the user, as they’ve defined it. Not necessarily a person’s name. - **leave**
3. **screen_name** - object - 585 - The screen name, handle, or alias that this user identifies themselves with. screen_names are unique but subject to change. Use id_str as a user identifier whenever possible. - **leave**
4. **protected** - bool - 2 - When true, indicates that this user has chosen to protect their Tweets. - **feature**
5. **followers_count** - int64 - 32 - The number of followers this account currently has. - **feature**
6. **fast_followers_count** - int64 - 1 -  - **drop**
7. **normal_followers_count** - int64 - 32 -  - **drop**
8. **friends_count** - int64 - 66 - The number of users this account is following (AKA their “followings”). - **feature**
9. **listed_count** - int64 - 11 - The number of public lists that this user is a member of.  - **feature**
10. **created_at** - object - 585 - The UTC datetime that the user account was created on Twitter. - **feature**
11. **favourites_count** - int64 - 80 - The number of Tweets this user has liked in the account’s lifetime. - **feature**
12. **geo_enabled** - bool - 1 - This field must be true for the current user to attach geographic data when using POST statuses / update - **feature**
13. **verified** - bool - 1 - When true, indicates that the user has a verified account. - **feature**
14. **statuses_count** - int64 - 276 - The number of Tweets (including retweets) issued by the user. - **feature**
15. **media_count** - int64 - 31 -  - **feature**
16. **contributors_enabled** - bool - 1 - deprecated - **drop**
17. **is_translator** - bool - 1 - deprecated - **drop**
18. **is_translation_enabled** - bool - 1 - deprecated - **drop**
19. **profile_background_color** - object - 1 - deprecated - **drop**
20. **profile_background_tile** - bool - 1 - deprecated - **drop**
21. **profile_image_url** - object - 569 - deprecated - **drop**
22. **profile_image_url_https** - object - 569 - deprecated - **drop**
23. **profile_link_color** - object - 1 - deprecated - **drop**
24. **profile_sidebar_border_color** - object - 1 - deprecated - **drop**
25. **profile_sidebar_fill_color** - object - 1 - deprecated - **drop**
26. **profile_text_color** - int64 - 1 - deprecated - **drop**
27. **profile_use_background_image** - bool - 1 - deprecated - **drop**
28. **has_extended_profile** - bool - 1 - deprecated - **drop**
29. **default_profile** - bool - 1 - When true, indicates that the user has not altered the theme or background of their user profile. - **drop**
30. **default_profile_image** - bool - 2 - When true, indicates that the user has not uploaded their own profile image and a default image is used instead.  - **feature**
31. **pinned_tweet_ids** - object - 23 - Unique identifier of this user's pinned Tweet. - **drop**
32. **pinned_tweet_ids_str** - object - 23 - Unique identifier of this user's pinned Tweet. - **drop**
33. **has_custom_timelines** - bool - 2 -  - **feature**
34. **advertiser_account_type** - object - 1 -  - **feature**
35. **advertiser_account_service_levels** - object - 1 -  - **drop**
36. **business_profile_state** - object - 1 -  - **drop**
37. **translator_type** - object - 1 - deprecated - **drop**
38. **withheld_in_countries** - object - 1 - Provides a list of countries where this user is not available. - **drop**
39. **require_some_consent** - bool - 1 -  - **drop**
40. **entities.description.urls** - object - 1 - Represents URLs included in the text of a Tweet. - **drop**
41. **status.created_at** - object - 583 - UTC time when this Tweet was created. - **drop**
42. **status.id** - float64 - 583 - The integer representation of the unique identifier for this Tweet. - **drop**
43. **status.id_str** - float64 - 583 - The string representation of the unique identifier for this Tweet. - **drop**
44. **status.text** - object - 582 - The actual UTF-8 text of the status update. - **drop**
45. **status.truncated** - object - 2 - Indicates whether the value of the text parameter was truncated, for example, as a result of a retweet exceeding the original Tweet text length limit of 140 characters. - **drop**
46. **status.entities.hashtags** - object - 1 - Represents hashtags which have been parsed out of the Tweet text. - **drop**
47. **status.entities.symbols** - object - 1 - Represents symbols, i.e. $cashtags, included in the text of the Tweet. - **drop**
48. **status.entities.user_mentions** - object - 262 - Represents other Twitter users mentioned in the text of the Tweet. - **drop**
49. **status.entities.urls** - object - 179 - Represents URLs included in the text of a Tweet. - **drop**
50. **status.source** - object - 1 - Utility used to post the Tweet, as an HTML-formatted string. Tweets from the Twitter website have a source value of web. - **drop**
51. **status.is_quote_status** - object - 2 - Indicates whether this is a Quoted Tweet.  - **drop**
52. **status.retweet_count** - float64 - 10 - Number of times this Tweet has been retweeted. - **drop**
53. **status.favorite_count** - float64 - 7 - Indicates approximately how many times this Tweet has been liked by Twitter users. - **drop**
54. **status.favorited** - object - 1 - Indicates whether this Tweet has been liked by the authenticating user. - **drop**
55. **status.retweeted** - object - 1 - Indicates whether this Tweet has been Retweeted by the authenticating user. - **drop**
56. **status.lang** - object - 6 - When present, indicates a BCP 47 language identifier corresponding to the machine-detected language of the Tweet text, or und if no language could be detected. - **drop**

After some research I decised to go with a list of 13 features:
```
numerical = ['statuses_count',
             'followers_count',
             'friends_count',
             'favourites_count',
             'listed_count',
             'media_count',
             'life_span']

categorical = ['default_profile_image',
              'geo_enabled',
              'protected',
              'verified',
              'has_custom_timelines',
              'advertiser_account_type']
```

## Model selection and parameter tuning 
I have trained LogisticRegression, DecisionTree, RandomForest and XGBoost.

The validation framework was created by the splitting dataset into train, validation and test sets. The target variable was removed from the original dataset.
The preprocessing is done using the DictVectorizer to make a one-hot encode for the categorical variable of type. 

The four models were trained and tuned on the following parameters:

**LogisticRegression**:
* C parameter of regularization

LogisticRegression(C=0.01, max_iter=1000, random_state=1) with roc_auc_score 96.43%.

**DecisionTree**:
* max_depth
* min_sample_leaf
DecisionTreeClassifier(max_depth=4, min_samples_leaf=5) with roc_auc_score 97.24%

**RandomForest**:
* n_estimators 
* max_depth 
* min_samples_leaf
RandomForestClassifier(max_depth=10, n_estimators=200, random_state=1) with roc_auc_score 98.90%

**XGBoost**
* eta
* max_depth
* min_child_weight
    'eta': 0.1, 
    'max_depth': 5,
    'min_child_weight': 1 
     with roc_auc_score = 98.74%

5.- The models were compared, where XGboost was the one with the best roc_auc_score

6.- The final model was trained with the full data train and compared with the test data, the roc_auc_score was better with a 98.24% of roc_auc_score.

## Deployment locally
