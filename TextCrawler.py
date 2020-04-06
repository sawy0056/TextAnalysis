

import time
import random
import urllib
#from six.moves import urllib
import urllib3
from bs4 import BeautifulSoup
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sqlalchemy import create_engine, MetaData, Table, select
import pyodbc
import re
import string
from collections import Counter

#For multi-threading
from joblib import Parallel, delayed
import multiprocessing


use_proxies = True
############################## PROXIES
proxy_file = 'C:\\Users\\sawy0\\Documents\\GitHub\\AMATH582\\FinalProject\\ProxyList.txt'
#proxy_link = 'https://api.proxyscrape.com/?request=getproxies&proxytype=http&timeout=10000&country=all&ssl=all&anonymity=all'
proxy_link = 'https://api.proxyscrape.com/?request=getproxies&proxytype=http&timeout=10000&country=all&ssl=all&anonymity=elite'

#Retrieve latest proxy list and store in dataframe
urllib.request.urlretrieve (proxy_link, proxy_file)
proxies = pd.read_csv(proxy_file, sep=" ", header=None)
proxies.columns = ["IP"]
proxies = proxies.sample(frac=1)


############################## USER AGENTS
agent_file = 'C:\\Users\\sawy0\\Documents\\GitHub\\AMATH582\\FinalProject\\UserAgents.txt'
agents = pd.read_csv(agent_file, sep="|", header=None)
agents = agents.drop(agents.columns[[1]],axis=1)
agents.columns = ["User_Agent"]
agents = agents.sample(frac=1)


############################## SETTINGS
#####The next two lines are needed to initialize your environment the first time
nltk.download('stopwords')
nltk.download('punkt')

#Parameters to connect to local SQL instance
server = 'DESKTOP-BR9P1E0\SQLEXPRESS'
database = 'TextAnalysis'
dbTable = 'MonthlyFreqData'

#Used in BeautifulSoup to locate news articles    
key1 = 'https://finance.yahoo.com/news'
key2 = 'http://finance.yahoo.com/news'

#out_dir = 'C:\\Users\\sawy0\\Desktop\\UW\\AMATH 582\\Final Project\\'
out_dir = "C://Users//sawy0//Documents//GitHub//AMATH582//FinalProject//"

#num_cores = multiprocessing.cpu_count()
num_cores = 100
numdays = 100
start_date = datetime.date(2012,1,4)
base = datetime.date(2013,1,1)
#base = datetime.datetime.today()
date_list = [base + datetime.timedelta(days=x) for x in range(numdays)]


#while end_date > start_date:
    
#Finds URLs, scrapes text, normalizes the text, counts freq, outputs to SQL
def TextCrawler(start_date): 
    failures = list()
    monthlyFreq = pd.DataFrame()
    df = pd.DataFrame(columns = ['Date','URL'])
    flag=0
    url = 'https://finance.yahoo.com/sitemap/' + str(start_date).replace("-","_") +'/'
    articles = [];

    while flag < 1:
        
        # Random Proxy                
        Pidx = random.randint(0,len(proxies.index)-1)
        proxy = 'http://' + proxies.loc[Pidx][0] + '/'
        #Random User Agent
        Aidx = random.randint(0,len(agents.index)-1)
        user_agent = {'user-agent': agents.loc[Aidx][0]}
        
        r = None
        while r is None:
            try:
                if use_proxies:
                    #pmanager = urllib3.ProxyManager(proxy, headers=user_agent, timeout=30)
                    pmanager = urllib3.ProxyManager(proxy, headers=user_agent)
                    r = pmanager.request('GET', url)
                else:
                    pmanager = urllib3.PoolManager()
                    r = pmanager.request('GET', url)
            except:
                failures.append(url)
                pass        
        tmp = r.data.decode('utf-8')
        soup = BeautifulSoup(tmp)

        for a in soup.find_all('a', href=True):
            if (key1 in a['href'] or key2 in a['href']) and a['href']!='https://finance.yahoo.com/news/' and 'edited-transcript' not in a['href']:
                df = df.append(pd.Series([start_date, a['href']], index = df.columns), ignore_index=True)
                articles.append(a['href'])
                #i=i+1
        
        #Find the next URL (ieif you navigated to thenext 50 articles by clicking next)
        for a in soup.find_all('a', href=True):
            if 'start' in a['href']:
                next_url = a['href']
                url = next_url
                flag = 0
            else:
                flag = 1 #When flag is 1, move next day
        #del http_pool
    
#    #Iterate to the next date
#    start_date = start_date+datetime.timedelta(days=1)

    #Create Output Directory if Needed
    Path(out_dir+'URL_Lists\\').mkdir(parents=True, exist_ok=True)

    urlfile = out_dir+'URL_Lists\\'+str(start_date).replace("-","_")+"_URL_List.txt"
    text_file = open(urlfile, "w")
    text_file.write(str(articles))
    text_file.close()

##################################################################################################
################# At this point we have df containing dates and article URLs
################# Need to now loop and scrape text/freq
##################################################################################################
    text_data = list()
    for i in range(len(df)):
        text_data = list()
#        ### Connect to the URL
#        x = df['URL'].loc[i]
#        http = urllib3.PoolManager()
#        r = http.request('GET', x)
#        tmp = r.data.decode('utf-8')
#        soup = BeautifulSoup(tmp)
        
        x = df['URL'].loc[i]
        r = None
        z=1
        while r is None:   
            z=z+1
            t_end = time.time() + 30
            try:        
                while time.time() < t_end:
                    if use_proxies:
                        ### Connect to the URL        
                        # Need a random Proxy
                        Pidx = random.randint(0,len(proxies.index)-1)
                        proxy = 'http://' + proxies.loc[Pidx][0] + '/'
                        # Need a random user agent        
                        Aidx = random.randint(0,len(agents.index)-1)
                        user_agent = {'user-agent': agents.loc[Aidx][0]}
                        #Create the proxymanager pool object and retrieve        
                        pmanager = urllib3.ProxyManager(proxy, headers=user_agent, timeout=120)
                        r = pmanager.request('GET', x)
                    else:
                        pmanager = urllib3.PoolManager()
                        r = pmanager.request('GET', x)
                    t_end = time.time()
            except:
                failures.append(x)
                pass  
            if z==10:
                r=1
                
        tmp = r.data.decode('utf-8')
        soup = BeautifulSoup(tmp)
        artsoup = soup.find('article')
        
        if artsoup is not None:
            
            #kill all script and style elements
            for script in artsoup(["script", "style"]):
                script.extract()    # rip it out
            
            text = artsoup.get_text()                                      #get text
            lines = (line.strip() for line in text.splitlines())        #break into lines and remove leading and trailing space on each        
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))  #break multi-headlines into a line each
            text = '\n'.join(chunk for chunk in chunks if chunk)        #drop blank lines
            text_data.append(text)                                      #Append to text_data object
            
            #Write the raw text
            filename = x
            filename = filename.replace('http://finance.yahoo.com/news/','')
            filename = filename.replace('https://finance.yahoo.com/news/','')
            filename = filename.replace('.html','')
            file = out_dir+'Articles\\'+str(start_date)+'\\'+filename+'.txt'
            
            #Create Output Directory if Needed
            Path(out_dir+'Articles\\'+str(start_date)+'\\').mkdir(parents=True, exist_ok=True)
            
    #        #Write the RAW article to a text file
            try:
                text_file = open(file, "w")
                text_file.write(text)
                text_file.close()    
            except:
                failures.append(x)
                pass
                
            ####################################################
            #Text Normalization, Corpus, and Frequency Block
            ####################################################        
            #https://medium.com/@datamonsters/text-preprocessing-in-python-steps-tools-and-examples-bf025f872908            
            text2 = text.lower()                #Make all lower case
            text2 = re.sub(r'\d+', '', text2)   #Remove #s        
            text2 = text2.translate(str.maketrans("","", string.punctuation))   #Remove Punctuation        
            #text2 = text2.replace("‘", "")      #Remove apostrophes
            #text2 = text2.replace("’", "")      #Remove apostrophes
            text2 = text2.replace('“', "")      #Remove quotes
            text2 = text2.replace('”', "")      #Remove quotes
            text2 = text2.replace("-", "")      #Remove dashes
            text2 = text2.replace("–", "")      #Remove dashes
#            text2 = text2.replace("\\", "")     #Remove dashes
            text2 = text2.replace("(", "")      #Remove dashes            
            text2 = text2.replace(")", "")      #Remove dashes            
            text2 = text2.strip()               #Remove White Spaces
            
            #Remove stopwords (and, the, but, etc...)
            stop_words = set(stopwords.words('english'))
            tokens = word_tokenize(text2)
            text2 = [i for i in tokens if not i in stop_words]
            
            if text2!=[]:
                
                #Get unique word list and frequencies
                dict_sum = Counter(text2)
                
                #Convert to dataframe
                out_df = pd.DataFrame(dict_sum.items(), columns=['Word', 'Freq'])            
                out_df['Date'] = str(df['Date'].loc[i])
                out_df['UpdateTime'] = str(datetime.datetime.utcnow())        
                #out_df = out_df.loc[out_df['Freq']>2]            
                out_df = out_df[['Date','Word','Freq','UpdateTime']]
                
                #Handle the monthly aggregates    
                monthlyFreq = monthlyFreq.append(out_df)
                
    monthlyFreq = monthlyFreq.groupby(['Word']).sum()
    monthlyFreq['Date'] = str(df['Date'].loc[i])
    monthlyFreq['UpdateTime'] = str(datetime.datetime.utcnow())     
    monthlyFreq = monthlyFreq.reset_index()
    monthlyFreq = monthlyFreq[['Date','Word','Freq','UpdateTime']]    
    monthlyFreq = monthlyFreq.loc[monthlyFreq['Freq']>1]

    #Write dataframe to SQL                        
    params = urllib.parse.quote_plus("DRIVER={SQL Server};SERVER="+server+";DATABASE="+database)
    engine = create_engine("mssql+pyodbc:///?odbc_connect=%s" % params) 
    cnxn = engine.connect() 
    monthlyFreq.to_sql(name=dbTable,con=cnxn, index=False, if_exists='append', method='multi', chunksize=100)            
    cnxn.close()
    engine.dispose()

    del df
    return 1



#########################
###  THREADING
#########################

#Thread Settings    
num_cores = 150
numdays = 150
base = datetime.date(2013,12,16)
date_list = [base + datetime.timedelta(days=x) for x in range(numdays)]


#Multi-Thread all of the above by date   
from joblib import parallel_backend
with parallel_backend('threading', n_jobs=num_cores):    
    results = Parallel(n_jobs=num_cores)(delayed(TextCrawler)(i) for i in date_list)    
    
    


#with parallel_backend('multiprocessing', n_jobs=num_cores):
#    results = Parallel(n_jobs=num_cores)(delayed(TextCrawler)(i) for i in date_list)    

           
    
