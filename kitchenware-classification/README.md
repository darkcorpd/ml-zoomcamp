# Kitchenware Classification
This image classification project corresponds to [MLZoomcamp 2022](http://mlzoomcamp.com/) Capstone-II Project.

## Problem Description

The classification of kitchenware has a variety of applications ranging from assisting physically disabled individuals to helping in daily household chores. A hierarchical classifier recognizes 10 classes of cutlery with a total of 9370 images that vary from class to class. Given the amount of noise and shape variations present in the segmented images and features, it was challenging to achieve a high accuracy rate for the given set of classes. Finally, an average accuracy of 90% was achieved with some improvements.

With the growth and advent of technology in the 21st century, every piece of human work is getting automated. This kind of automation has reduced human effort and enhanced their focus on more difficult and challenging jobs instead of the day to day menial jobs. The way technology and robotics has entered our homes is astonishing. This trend is growing exponentially and will continue to grow with young researchers and engineers pioneering their ideas in this field. From automatic lights, fans, heaters and other electronic devices, imagine a robot helping you in the kitchen to cook a delicious meal whilst you sit leisurely. This robot could also be helpful in washing the dishes, lending you a helping hand while you are cooking, cleaning the table after the meal is done and also preparing the whole meal.

The idea behind this work is to improve the effectiveness of the classifier, deploy locally and prepare a service for cloud deployment.

## Dataset Description

The dataset is from the [Kitchenware Classification](https://www.kaggle.com/competitions/kitchenware-classification/) Kaggle contest organized by [DataTalks.Club](https://datatalks.club).

![image](https://user-images.githubusercontent.com/91184329/213883475-c97a918f-efc0-437e-83f2-51c2773f7832.png)

Toloka was used for collecting the images for this competition. If you want to know how to collect a similar dataset, check this workshop: https://www.youtube.com/watch?v=POGiLFWxQWQ

The dataset contains 9370 images in .jpg format of different kitchenware, total size 1.75 GB.

The goal of the contest is to classify images of different kitchenware items into 6 classes:
* cups
* glasses
* plates
* spoons
* forks
* knives

## Project Setup

[Saturn Cloud](https://saturncloud.io) platform was used for data preparation and model training. 

A starter notebook for the Kitchenware classification competition on Kaggle: [keras-starter.ipynb](https://github.com/DataTalksClub/kitchenware-competition-starter/blob/main/keras-starter.ipynb) is a good start to learn how to:

* Download the data from Kaggle and unzip it
* Read the data
* Train an xception model
* Make predictions
* Submit the results

You can run this notebook in SaturnCloud:

<p align="center">
    <a href="https://app.community.saturnenterprise.io/dash/resources?recipeUrl=https://raw.githubusercontent.com/DataTalksClub/kitchenware-competition-starter/main/kitchenware-jupyter-recipe.json" target="_blank" rel="noopener">
        <img src="https://saturncloud.io/images/embed/run-in-saturn-cloud.svg" alt="Run in Saturn Cloud"/>
    </a>
</p>

See more on [GitHUB](https://github.com/DataTalksClub/kitchenware-competition-starter).

## Running locally with gunicorn/waitress

1. Clone this repository on your computer.

2. Install dependencies from `Pipfile` by running command:

```sh
pipenv install
```
3. Activate virtual environment:

```sh
pipenv shell
```
4. Run service with gunicorn:

```sh
pipenv run gunicorn --bind 0.0.0.0:9696 predict:app
```
Or with waitress:

```sh
waitress-serve --listen=0.0.0.0:9696 predict:app
```
5. Run [test.py](https://github.com/darkcorpd/ml-zoomcamp/blob/main/kitchenware-classification/test.py) to see attrition prediction on given data.

![image](https://user-images.githubusercontent.com/91184329/213935382-b84cb16f-785a-4b6b-9d8f-428df5281f8b.png)


## Running locally with Docker

1. Build an image from a Dockerfile by the command:
 
```sh
docker build -t kitchenware-classification .
```

2. Run service:

```sh
docker run --rm -it -p 9060:9060 -d  kitchenware-classification
```

3. Run test.py to see probabilities of which class the kitchenware belongs to.

The result of prediction:

4. Change the image in test.py and see the new prediction.

## TensorFlow-Serving

TensorFlow Serving (`tf-serving`) requires models to be in a specific format.

```python
import tensorflow as tf
from tensorflow import keras
model = keras.models.load_model('model.h5')
tf.saved_model.save(model, 'tlite_model')
```

This code loads a Keras model and simply saves it to Tensorflow format rather than Keras.

Tensorflow-formatted models are directories containing a number of files:

```
converted_model
┣╸ assets
┣╸ fingerprint.pb
┣╸ saved_model.pb
┗╸ variables
    ┣╸ variables.data-00000-of-00001
    ┗╸ variables.index
```

The `saved_model_cli` utility allows us to inspect the contents of the tf model:
* `saved_model_cli show --dir converted_model --all`

The ***signature definition*** describes both the inputs and the outputs of the model:

    signature_def['serving_default']:
        The given SavedModel SignatureDef contains the following input(s):
            inputs['input_4'] tensor_info:
                dtype: DT_FLOAT
                shape: (-1, 299, 299, 3)
                name: serving_default_input_8:0
        The given SavedModel SignatureDef contains the following output(s):
            outputs['dense_3'] tensor_info:
                dtype: DT_FLOAT
                shape: (-1, 10)
                name: StatefulPartitionedCall:0
        Method name is: tensorflow/serving/predict

We can run the tf-serving Docker image with model mounted in a volume with the following command:

```sh
docker run -it --rm \
    -p 8500:8500 \
    -v "$(pwd)/tlite_model:/models/tlite_model/1" \
    -e MODEL_NAME="tlite_model" \
    tensorflow/serving:2.7.0
```

You can test the tf-serving container with Jupyter Notebook ([gateway.ipynb](https://github.com/darkcorpd/ml-zoomcamp/blob/main/kitchenware-classification/gateway.ipynb)):

![image](https://user-images.githubusercontent.com/91184329/214132828-46472f09-20d1-4dfc-af4d-fb714eb923a8.png)

To export a Jupyter Notebook to a Python script use the following command:

```sh
jupyter nbconvert --to script gateway.ipynb
```

## Running everything locally with docker-compose

`image-model.dockerfile` is used to dockerize our model: 

```sh
docker build -t kitchen-model:v1 -f image-model.dockerfile .
```
We can now run this new image like this:

```sh
docker run -it --rm \
    -p 8500:8500 \
    kitchen-model:v1
```

`image-gateway.dockerfile` is used to dockerize our gateway:

```sh
docker build -t kitchen-gateway:v1 -f image-gateway.dockerfile .
```

And now run it:

```sh
docker run -it --rm \
    -p 9696:9696 \
    kitchen-gateway:v1
```
With both images running, we can now test them with a simple `test.py` script.

We can now run the app like this:

```sh
docker-compose up
```

We can shut down the app with:

```sh
docker-compose down
```

## Deploying TensorFlow models to Kubernetes

Load the image to Kind, apply the deployment to cluster, test it by forwarding the pod's port and using the local gateway script:

```sh
kind load docker-image kitchen-model:v1
kubectl apply -f model-deployment.yaml
kubectl get pod
kubectl port-forward tf-serving-kitchen-model-#add_here_the_id# 8500:8500
# call the gateway script after the command above from another terminal
```

Create TF-Serving model deployment service, test it with port-forwarding and the local gateway script:

```sh
kubectl apply -f model-service.yaml
kubectl get pod
kubectl port-forward service/tf-serving-kitchen-model 8500:8500
# call the gateway script after the command above from another terminal
```

Deploying the Gateway, repeating the same steps we did for the model, but this time for the gateway.

```sh
kind load docker-image kitchen-gateway:v2
kubectl apply -f gateway-deployment.yaml
kubectl get pod
kubectl port-forward gateway-#add_here_the_id# 9696:9696
# Now run the test script to test the gateway
```

Create Gateway deployment service

```sh
kubectl apply -f gateway-service.yaml
kubectl get service
kubectl port-forward service/gateway 8080:80
# Now run the test script to test the gateway service
```
