import requests
from elasticsearch import Elasticsearch as ES
import json
import csv
import time
import pandas as pd
import pandas as pd
import numpy as np

from keras.models import Model, load_model
from sklearn.metrics import accuracy_score
import joblib 

def pull_one_line():
    es=ES([{'host':'192.168.1.112','port':9200}], timeout = 15)
    es
    i = 0
    output = []
    while(i < 1):

        res = es.search(index='metricbeat-*', body={
		'query':{
		    'dis_max': {
			"queries": [            
			{'match':{'host.id':'d57d409de59c4e9fbe9adb2629121b07'}},
		        {'match': { 'metricset.name':'process'}}],
			
		     }
		},
		"size": 5,
		"sort" : [
		{"@timestamp": {"order": "desc"}},
		{"_score": {"order": "desc"}}
		
		  
		]
	    })
        es

        webserver1 = res['hits']['hits'][0]
        webserver2 = res['hits']['hits'][1]
        webserver3 = res['hits']['hits'][2]
        webserver4 = res['hits']['hits'][3]
        webserver5 = res['hits']['hits'][4]

        res = es.search(index='metricbeat-*', body={
		'query':{
		    'dis_max': {
			"queries": [            
			{'term':{'host.id':'ec2e1b323d7d840fcded7f9bc879a32d'}},
		        {'term': { 'metricset.name':'process'}}
		        
			],
			
		     }
		},
		"size": 5,
		"sort" : [
		{"@timestamp": {"order": "desc"}},
		{"_score": {"order": "desc"}}
		
		  
		]
	    })
        es

        database1 = res['hits']['hits'][0]
        database2 = res['hits']['hits'][1]
        database3 = res['hits']['hits'][2]
        database4 = res['hits']['hits'][3]
        database5 = res['hits']['hits'][4]


        #print(database)

        res = es.search(index='filebeat-*', body={
		'query':{
		    'dis_max': {
			"queries": [            
			{'match':           {'host.id':'d57d409de59c4e9fbe9adb2629121b07'}},
		        {'match': { 'fileset.name':'access'}}],
			
		     }
		},
		"size": 5,
		"sort" : [
		{"@timestamp": {"order": "desc"}},
		{"_score": {"order": "desc"}}
		
		  
		]
	    })

        es
    
        access1 = res['hits']['hits'][0]
        access2 = res['hits']['hits'][1]
        access3 = res['hits']['hits'][2]
        access4 = res['hits']['hits'][3]
        access5 = res['hits']['hits'][4]

        #print(access)
    
        output.append([])
        output.append([])
        output.append([])
        output.append([])
        output.append([])

        output[0].append(webserver1['_source']['system']['process']['memory']['size'])
        output[0].append(webserver1['_source']['system']['process']['memory']['rss']['bytes'])
        output[0].append(webserver1['_source']['system']['process']['memory']['rss']['pct'])
        output[0].append(webserver1['_source']['system']['process']['memory']['share'])
        output[0].append(webserver1['_source']['system']['process']['cpu']['total']['value'])
        output[0].append(webserver1['_source']['system']['process']['cpu']['total']['pct'])
        output[0].append(webserver1['_source']['system']['process']['cpu']['total']['norm']['pct'])

        output[0].append(database1['_source']['system']['process']['memory']['size'])
        output[0].append(database1['_source']['system']['process']['memory']['rss']['bytes'])
        output[0].append(database1['_source']['system']['process']['memory']['rss']['pct'])
        output[0].append(database1['_source']['system']['process']['memory']['share'])
        output[0].append(database1['_source']['system']['process']['cpu']['total']['value'])
        output[0].append(database1['_source']['system']['process']['cpu']['total']['pct'])
        output[0].append(database1['_source']['system']['process']['cpu']['total']['norm']['pct'])

        output[0].append(access1['_source']['url']['original'])
        output[0].append(access1['_source']['http']['request']['method'])
        output[0].append(access1['_source']['http']['response']['status_code'])
        output[0].append(access1['_source']['http']['response']['body']['bytes'])
        
        output[0].append(access1['_source']['user_agent']['name'])
        #output[0].append(access1['_source']['user_agent']['device']['name'])
       

        output[1].append(webserver1['_source']['system']['process']['memory']['size'])
        output[1].append(webserver1['_source']['system']['process']['memory']['rss']['bytes'])
        output[1].append(webserver1['_source']['system']['process']['memory']['rss']['pct'])
        output[1].append(webserver1['_source']['system']['process']['memory']['share'])
        output[1].append(webserver1['_source']['system']['process']['cpu']['total']['value'])
        output[1].append(webserver1['_source']['system']['process']['cpu']['total']['pct'])
        output[1].append(webserver1['_source']['system']['process']['cpu']['total']['norm']['pct'])

        output[1].append(database1['_source']['system']['process']['memory']['size'])
        output[1].append(database1['_source']['system']['process']['memory']['rss']['bytes'])
        output[1].append(database1['_source']['system']['process']['memory']['rss']['pct'])
        output[1].append(database1['_source']['system']['process']['memory']['share'])
        output[1].append(database1['_source']['system']['process']['cpu']['total']['value'])
        output[1].append(database1['_source']['system']['process']['cpu']['total']['pct'])
        output[1].append(database1['_source']['system']['process']['cpu']['total']['norm']['pct'])

        output[1].append(access2['_source']['url']['original'])
        output[1].append(access2['_source']['http']['request']['method'])
        output[1].append(access2['_source']['http']['response']['status_code'])
        output[1].append(access2['_source']['http']['response']['body']['bytes'])
        
        output[1].append(access2['_source']['user_agent']['name'])
        #output[1].append(access2['_source']['user_agent']['device']['name'])
      
        output[2].append(webserver1['_source']['system']['process']['memory']['size'])
        output[2].append(webserver1['_source']['system']['process']['memory']['rss']['bytes'])
        output[2].append(webserver1['_source']['system']['process']['memory']['rss']['pct'])
        output[2].append(webserver1['_source']['system']['process']['memory']['share'])
        output[2].append(webserver1['_source']['system']['process']['cpu']['total']['value'])
        output[2].append(webserver1['_source']['system']['process']['cpu']['total']['pct'])
        output[2].append(webserver1['_source']['system']['process']['cpu']['total']['norm']['pct'])

        output[2].append(database1['_source']['system']['process']['memory']['size'])
        output[2].append(database1['_source']['system']['process']['memory']['rss']['bytes'])
        output[2].append(database1['_source']['system']['process']['memory']['rss']['pct'])
        output[2].append(database1['_source']['system']['process']['memory']['share'])
        output[2].append(database1['_source']['system']['process']['cpu']['total']['value'])
        output[2].append(database1['_source']['system']['process']['cpu']['total']['pct'])
        output[2].append(database1['_source']['system']['process']['cpu']['total']['norm']['pct'])

        output[2].append(access3['_source']['url']['original'])
        output[2].append(access3['_source']['http']['request']['method'])
        output[2].append(access3['_source']['http']['response']['status_code'])
        output[2].append(access3['_source']['http']['response']['body']['bytes'])
        
        output[2].append(access3['_source']['user_agent']['name'])
        #output[2].append(access3['_source']['user_agent']['device']['name'])
       
        output[3].append(webserver1['_source']['system']['process']['memory']['size'])
        output[3].append(webserver1['_source']['system']['process']['memory']['rss']['bytes'])
        output[3].append(webserver1['_source']['system']['process']['memory']['rss']['pct'])
        output[3].append(webserver1['_source']['system']['process']['memory']['share'])
        output[3].append(webserver1['_source']['system']['process']['cpu']['total']['value'])
        output[3].append(webserver1['_source']['system']['process']['cpu']['total']['pct'])
        output[3].append(webserver1['_source']['system']['process']['cpu']['total']['norm']['pct'])
        output[3].append(database1['_source']['system']['process']['memory']['size'])
        output[3].append(database1['_source']['system']['process']['memory']['rss']['bytes'])
        output[3].append(database1['_source']['system']['process']['memory']['rss']['pct'])
        output[3].append(database1['_source']['system']['process']['memory']['share'])
        output[3].append(database1['_source']['system']['process']['cpu']['total']['value'])
        output[3].append(database1['_source']['system']['process']['cpu']['total']['pct'])
        output[3].append(database1['_source']['system']['process']['cpu']['total']['norm']['pct'])
        output[3].append(access4['_source']['url']['original'])
        output[3].append(access4['_source']['http']['request']['method'])
        output[3].append(access4['_source']['http']['response']['status_code'])
        output[3].append(access4['_source']['http']['response']['body']['bytes'])
        
        output[3].append(access4['_source']['user_agent']['name'])
        #output[3].append(access4['_source']['user_agent']['device']['name'])
       

        output[4].append(webserver1['_source']['system']['process']['memory']['size'])
        output[4].append(webserver1['_source']['system']['process']['memory']['rss']['bytes'])
        output[4].append(webserver1['_source']['system']['process']['memory']['rss']['pct'])
        output[4].append(webserver1['_source']['system']['process']['memory']['share'])
        output[4].append(webserver1['_source']['system']['process']['cpu']['total']['value'])
        output[4].append(webserver1['_source']['system']['process']['cpu']['total']['pct'])
        output[4].append(webserver1['_source']['system']['process']['cpu']['total']['norm']['pct'])
        output[4].append(database1['_source']['system']['process']['memory']['size'])
        output[4].append(database1['_source']['system']['process']['memory']['rss']['bytes'])
        output[4].append(database1['_source']['system']['process']['memory']['rss']['pct'])
        output[4].append(database1['_source']['system']['process']['memory']['share'])
        output[4].append(database1['_source']['system']['process']['cpu']['total']['value'])
        output[4].append(database1['_source']['system']['process']['cpu']['total']['pct'])
        output[4].append(database1['_source']['system']['process']['cpu']['total']['norm']['pct'])

        output[4].append(access5['_source']['url']['original'])
        output[4].append(access5['_source']['http']['request']['method'])
        output[4].append(access5['_source']['http']['response']['status_code'])
        output[4].append(access5['_source']['http']['response']['body']['bytes'])
        
        output[4].append(access5['_source']['user_agent']['name'])
        #output[4].append(access5['_source']['user_agent']['device']['name'])
       

        i += 1
        


        df = pd.DataFrame(output)
        print(df)
        return(df)


def main():

   
    pull_one_line()
    
url_array = []
for i in range(1, 50):
  url_array.append('/inc/image.php?fileId='+str(i))
  url_array.append('/detail.php?projId='+str(i))
pages = ['/index.php', '/projects.php', '/about.php', '/services.php', '/support.php', '/contact.php', '/privacy.php']
for i in pages:
  url_array.append(i)
numbers = range(1, len(url_array)+1)
url_dictionary = dict(zip(url_array,numbers))

def url_encoder(s):
  if s == '*':
    return 128
  elif s in url_dictionary:
    return url_dictionary[s]
  else:
    return -1


dataframe1 = pull_one_line()
data_columns = ['apache_mem_size', 'apache_mem_bytes', 'apache_mem_percent',
       'apache_total_mem', 'cpu_value', 'curr_percentage', 'usual_percentage',
       'db_size', 'db_bytes', 'db_pct', 'db_total', 'db_value', 'db_curr_pct',
       'db_usual_pct', 'url', 'method', 'status_code', 'bytes', 'type']
dataframe1.columns = data_columns
for i in dataframe1.index:
  dataframe1.at[i, 'url'] = url_encoder(dataframe1.at[i, 'url'])
  cols  = ['method', 'type', 'status_code']
for x in cols:
  dataframe1 = pd.concat([dataframe1,pd.get_dummies(dataframe1[x])],axis=1)
  dataframe1.drop([x],axis=1, inplace=True)
columns1 = dataframe1.columns
required_columns = [   'apache_mem_size',   'apache_mem_bytes', 'apache_mem_percent',
         'apache_total_mem',          'cpu_value',    'curr_percentage',
         'usual_percentage',            'db_size',           'db_bytes',
                   'db_pct',           'db_total',           'db_value',
              'db_curr_pct',       'db_usual_pct',                'url',
                    'bytes',                'GET',            'OPTIONS',
                     'POST',              'Other',             'Spider',
                        200,                  404]
diff =  list(set(required_columns) - set(columns1))
for j in diff:
  dataframe1[j] = [ 0 for i in range(dataframe1.shape[0])]
test_data = dataframe1

model = load_model("project_150.h5")
scaler = joblib.load('scaler_150.pkl')

columns = test_data.columns

test_data = scaler.transform(test_data[columns])
labels_columns = [ 'normal' , 'fast_scan', 'spam', 'scan', '404_attack' ]

ypred = model.predict(test_data)

for i in ypred:
  index = np.argmax(i)
  print(labels_columns[index])

#main()
        
