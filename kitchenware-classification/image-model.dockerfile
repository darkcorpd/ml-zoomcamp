FROM tensorflow/serving:2.7.0

COPY tlite-model /models/tlite-model/1
ENV MODEL_NAME="tlite-model"