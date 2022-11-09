# Midterm project
**by Albert Tuykin**  
darkcorpd@gmail.com

for the [Machine Learning Zoomcamp](https://mlzoomcamp.com) by Alexey Grigorev.

## Twitter bot detection using supervised machine learning

![image](https://user-images.githubusercontent.com/91184329/200782202-6f61f9de-21df-43a9-ac85-45ab19eb2347.png)


In the world of Internet and social media, there are about 3.8 billion active social media users and half of the internet traffic consists of mostly bots.

Malicious bots make up 20% of the traffic, they are used for nefarious purposes, they can mimic human behavior, they can impersonate legal traffic, attack IoT devices and exploit their performance. 
Among all these concerns, the primary concern is for social media users as they represent a large group of active users on the internet, they are more vulnerable to breach of data, change in opinion based on data. Detection of such bots is crucial to prevent further mishaps. 

I use supervised Machine learning techniques in this project such as Logistic regression, Decision tree, Random Forest and XGBoost to calculate their accuracies and compare to detect Twitter bots from a collected training data set. 

## Data collection

I have used a [list](https://github.com/antibot4navalny/accounts_labelled/blob/main/labels.json) of Twitter accounts recognized as bots. This list is used in Chrome extention '[MetaBot for Twitter](https://github.com/antibot4navalny/metabot)' to highlight bots on Twitter.

I have used a `twitter_scraper_selenium` library to get profile details.

The list comparation is used to update the collected dataset by the information for new names in the bot list (updatable list).

The same way is used to collected data on real organic users provided by the same author `antibot4navalny`. Two datasets of bot and organic users' profiles are mixed to get a train set for a bot classification problem. The cleaned of duplicates dataset contains 1220 rows × 157 columns.

## Exploratory data analysis (EDA) 
is an especially important activity in the routine of a data analyst or scientist. It enables an in depth understanding of the dataset, define or discard hypotheses and create predictive models on a solid basis.

1. Remove duplicates.
2. Drop actually empty columns by the thereshold. To find a thereshold by which I will drop the columns I've plotted a distribution chart for NaN values. All the columns that have more than 2 NaN values can be removed.
3. Explore the data by the common approach:
*	**Variable:** name of the variable.
*	**Type:** the type or format of the variable. This can be categorical, numeric, Boolean, and so on.
*	**Context:** useful information to understand the semantic space of the variable. In the case of our dataset, the context is the social one.
*	**Expectation:** how relevant is this variable with respect to our task? We can use a scale “High, Medium, Low”.
*	**Comments:** whether or not we have any comments to make on the variable.

The discription of the variables can be found [here](https://developer.twitter.com/en/docs/twitter-api/v1/data-dictionary/object-model/user)

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
I have trained 4 models: LogisticRegression, DecisionTree, RandomForest and XGBoost.

The validation framework was created by the splitting dataset into train, validation and test sets. The target variable was removed from the original dataset.
The preprocessing is done using the DictVectorizer to make a one-hot encode for the categorical variable of type. 

The four models were trained and tuned on the following parameters:

**LogisticRegression**:
* C parameter of regularization
LogisticRegression(C=0.01, max_iter=1000, random_state=1)

**DecisionTree**:
* max_depth
* min_sample_leaf
DecisionTreeClassifier(max_depth=4, min_samples_leaf=5)

**RandomForest**:
* n_estimators 
* max_depth 
* min_samples_leaf
RandomForestClassifier(max_depth=10, n_estimators=200, random_state=1)

**XGBoost**:
* eta
* max_depth
* min_child_weight
XGBClassifier(eta=0.1, max_depth=5, min_child_weight=1)

The models were compared, where RandomForest was the one with the best roc_auc_score:

|Model|roc_auc_score|
|-----|--|
|LogisticRegression|96.43%|
|DecisionTree|97.24%|
|**RandomForest**|**98.90%**|
|XGBoost|98.74%|

The final model was trained with the full data train and compared with the test data, the roc_auc_score was 98.24%.

## Deployment locally using [BentoML](https://docs.bentoml.org/en/latest/tutorial.html)

The file [build_bento_model_twitter_bots.ipynb](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/8_week/build_bento_model_maintenance.ipynb) can be used to train the best model and save into a BentoML model for deployment.

The following command is used to import the BentoML model:

```bash
bentoml models import twitter_bots_predict_model:asw4gns4q2vxgjv5.bentomodel
```

The file [service.py](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/8_week/train.py) is elaborated with the script to deploy it locally using the BentoML interface. 

Local deployment can be run using the following command in the terminal:

```
bentoml serve service.py:svc --production
```

The Swagger interfase can be found by the address of [local host](http://localhost:3000)

## Deployment locally using Jupyter Notebook 

The file [predict.ipynb](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/8_week/predict.ipynb) can be used to load the BentoML model and predict if a Twitter user is bot or not. 

A dictionary scheme must be used:

```python
{"type": str,
 "air_temperature_[k]": float,
 "process_temperature_[k]": float,
 "rotational_speed_[rpm]": int,
 "torque_[nm]": float,
 "tool_wear_[min]": int}
 ```
 
 Here is an example
 
 ```python
 {"type": "L",
  "air_temperature_[k]": 298.0,
  "process_temperature_[k]": 308.7,
  "rotational_speed_[rpm]": 1268,
  "torque_[nm]": 69.4,
  "tool_wear_[min]": 189}
  ```
## Deployment using Docker

Once you create you bento model in the script [build_bento_model_maintenance.ipynb](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/8_week/build_bento_model_maintenance.ipynb), you need to create a [bentofile.yaml](https://github.com/FranciscoOrtizTena/ML_Zoomcamp/blob/main/8_week/bentofile.yaml), specifying the service, some labels, programming language and the different packages to use. 


Then you need to build your bento with.

```bash
bentoml build
```
  
To deploy your model using the Docker images, you need first to containerize the previous model into a Docker image, using the bentomodel file, type the following on your terminal to containerize it.


```bash
bentoml containerize maintenance_predict_classifier:fjxpm3s4soefsjv5
```

Once it's containerize it you can build the image using the following command on your terminal, remember to check the tag number for containerize it

```bash
docker run -it --rm -p 3000:3000 maintenance_predict_classifier:fjxpm3s4soefsjv5 serve --production
```

Another way is to download the docker image from the repository in the [docker hub](https://hub.docker.com/r/franciscoortiztena/maintenance_predict_classifier) 

First you need to download the docker image with the following command in the terminal

```bash
docker pull franciscoortiztena/maintenance_predict_classifier
```

And then run the following command

```bash
docker run -it --rm -p 3000:3000 franciscoortiztena/maintenance_predict_classifier serve --production
```

As the one in deploying, you can visit the [local host](http://0.0.0.0:3000/) to make the predictions

## Deployment using [Yandex Serverless Containers](https://cloud.yandex.com/en/docs/serverless-containers/operations)

1. Register, install and initialize the [Yandex Cloud command line interface](https://cloud.yandex.com/en/docs/cli/quickstart#install).

2. Create a container registry:

```
yc container registry create --name <my-reg>
```
Result:

```
done
id: crpd50616s9a2t7gr8mi
folder_id: b1g88tflru0ek1omtsu0
name: my-reg
status: ACTIVE
created_at: "2022-11-09T14:34:06.601Z"
```

Make sure the registry was created:

```
yc container registry list
```
Result:
|ID|NAME|FOLDER ID|
|-----|--|--|
|crpd50616s9a2t7gr8mi|my-reg|b1g88tflru0ek1omtsu0|


3. Create IAM token and authorize with IAM token:
```
yc iam create-token
```

```
docker login --username iam --password <IAM TOKEN> cr.yandex
```

4. Push container to the registry:
```
docker push <LONG_NAME>:<TAG>
```

5. Create a serverless container:
```
yc serverless container create --name <container_name>
```

Result:
```
id: bba3fva6ka5g********
folder_id: b1gqvft7kjk3********
created_at: "2021-07-09T14:49:00.891Z"
name: my-beta-container
url: https://bba3fva6ka5g********.containers.yandexcloud.net/
status: ACTIVE
```

6. Create a container revision with prepared image and [service account ID](https://cloud.yandex.com/en/docs/iam/operations/sa/get-id):
```
yc serverless container revision deploy \
  --container-name <container_name> \
  --image <Docker_image_URL> \
  --cores 1 \
  --memory 1GB \
  --concurrency 1 \
  --execution-timeout 30s \
  --service-account-id <service_account_ID>
```

Check revision by container name

```
yc serverless container revision list --container-name <container_name>
```

7. Make it public and get an invocation link

```
yc serverless container allow-unauthenticated-invoke <container_name>
```
```
yc serverless container get <container_name>
```
Result:

URL: <https://bbah254r1kim16a93724.containers.yandexcloud.net/>
