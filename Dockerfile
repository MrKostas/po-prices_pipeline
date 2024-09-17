FROM apache/airflow:2.7.3-python3.9.19
COPY requirements.txt ~/Desktop/po-prices_pipeline/requirements.txt

RUN pip install -U pip
RUN pip install -r ~/Desktop/po-prices_pipeline/requirements.txt