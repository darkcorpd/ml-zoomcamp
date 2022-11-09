import numpy as np
import bentoml
from bentoml.io import JSON
from pydantic import BaseModel

# Setting up data validation class
class profile_data(BaseModel):
    statuses_count: float
    followers_count: float
    friends_count: float
    favourites_count: float
    listed_count: float
    media_count: float
    life_span: int
    default_profile_image: bool
    geo_enabled: bool
    protected: bool
    verified: bool
    has_custom_timelines: bool
    advertiser_account_type: object

model_ref = bentoml.sklearn.get("twitter_bot_classify_model:xng2m3daog2ndodq")
dv = model_ref.custom_objects["dictVectorizer"]

model_runner = model_ref.to_runner()

svc = bentoml.Service("twitter_bot_classifier", runners=[model_runner])

@svc.api(input=JSON(pydantic_model=profile_data), output=JSON())
def twitter_bot_classify(profile_data):
    data = profile_data.dict()
    vector = dv.transform(data)
    prediction = model_runner.predict.run(vector)
    print(prediction)
    result = prediction[0]
    if result == 1:
        return {"status": "BOT"}
    elif result == 0:
        return {"status": "ORGANIC"}
    else:
        return {"status": "CHECK AGAIN"}