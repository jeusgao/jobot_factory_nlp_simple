A light tool for TF Model building/training/evaluating for simple nlp tasks just by params configuration, training/evaluating monitor and params configure GUI with streamlit.

copy pretrained LM into hub/bases/ (bert, rbt, albert ...)

## How to run:
1. run init_params first to generate params template...
2. streamlit run home.py --server.port=PORT
3. celery -A celery_training worker -l INFO --pidfile=celery/%n.pid --logfile=celery/%n%I.log

open localhost:PORT with a broswer ...

## Rest service:
* uvicorn rest_service.handlers:app --port=REST_SERVER-PORT or python3 rest_server.py -p=REST_SERVER-PORT

## ENV
python >= 3.6
 and test with tf 2.4.0

## references:
<a href="https://github.com/CyberZHG/keras-bert">keras-bert</a>
</br>
<a href="https://github.com/BrikerMan/Kashgari.git">Kashgari</a>
