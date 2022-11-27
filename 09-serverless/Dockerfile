FROM svizor42/zoomcamp-dino-dragon-lambda:v2

RUN pip install numpy
RUN pip install pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.7.0-cp39-cp39-linux_x86_64.whl

COPY hw_lambda_function.py .

CMD [ "hw_lambda_function.lambda_handler" ]