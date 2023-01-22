import requests

# Local deployment
url = 'http://localhost:9696/predict'

# Sample image
image_url = 'https://raw.githubusercontent.com/darkcorpd/ml-zoomcamp/main/kitchenware-classification/cup.jpg'
#image_url = 'https://storage.googleapis.com/kagglesdsdata/competitions/42532/4724339/images/0000.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20230122%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20230122T152034Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=4bfc372d64cf2605b99153730e434755dd246e4f70cd81ede5423a520ebe8166e12fb9482c7e019467f249d58239c0edf0e9b25c2408c58f697f044806a9dfacfe0a16d601b6de562fec3f10d805e00c2e5e195e245450b6f4e24d206c5452fc28c4ba15ef7396fae7dd211114d043929075a396057a701fb30d7b29c76b6fdfc3ef5ecb461f410c23b0b882b102c754a3563b7acca6c73efb40c2a37565ac1614acaa45f395e4698e086821dc67c310ae64c8592e236dbf3c2736f85f01bd2686595f0cfb3c3b9ec86beef2997a112d0067ab4fa7196c97b525b0457f01ffb24b5133d4d65b5b596f4a699fae859ba4ec7e572557189087511ee9f3ffd5e7d2'
#image_url = 'cup.jpg'

data = {'url': image_url}

result = requests.post(url, json=data).json()
print(result)