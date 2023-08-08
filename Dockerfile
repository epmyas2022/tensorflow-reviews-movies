FROM python:3.8-bullseye as base

#WORDIR

WORKDIR /usr/src/app


RUN pip install virtualenv

RUN virtualenv /venv
ENV PATH="/venv/bin:$PATH"

RUN pip install pandas && \
    pip install scikit-learn && \
    pip install "tensorflow<2.10" && \
    pip install matplotlib && \
    pip install gdown


#CREATE DIRECTORY data

RUN mkdir data && \
    cd data && \
    gdown --id 1iN4CfzhuXFWzMaplAbrzlJPKoYgIW30c


#DEFINIR VOLUMEN

VOLUME [ "/usr/src/app" ]  


CMD [ "python", "predict.py"]