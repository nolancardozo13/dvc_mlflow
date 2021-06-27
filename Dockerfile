# import base python image
FROM nvcr.io/nvidia/pytorch:21.05-py3

WORKDIR /mlflow

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# download data
RUN apt-get update && apt-get install -y \
    && apt-get install unzip 
COPY kaggle.json /root/.kaggle/kaggle.json
