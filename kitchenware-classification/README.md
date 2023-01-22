# Kitchenware Classification
This image classification project corresponds to [MLZoomcamp 2022](http://mlzoomcamp.com/) Capstone-II Project.


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

## Running locally with waitress

1. Clone this repository on your computer.

2. Install dependencies from `Pipfile` by running command:
```sh
pipenv install
```
3. Activate virtual environment:
```sh
pipenv shell
```
4. Run service with waitress:
```sh
waitress-serve --listen=0.0.0.0:9060 predict:app
```
5. Run test.py to see attrition prediction on given data.

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
