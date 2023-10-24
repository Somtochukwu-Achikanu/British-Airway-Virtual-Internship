import requests
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import numpy as np
import string
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
import re
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from collections import Counter
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer('english')

base_url = 'https://www.airlinequality.com/airline-reviews/british-airways'

pages = 37
page_size = 100
reviews=[]

#To loop through the pages 
for i in range(1, pages + 1):
    print(f"Scraping page{i}")
    
        # Create URL to collect links from paginated data
    url = f"{base_url}/page/{i}/?sortby=post_date%3ADesc&pagesize={page_size}"

    #collect html
    req = requests.get(url)  

    #Parse content
    content = req.content
    parsed_content = BeautifulSoup(content, 'html.parser')
    for para in parsed_content.find_all("div", {"class":"text_content"}):
        reviews.append(para.get_text())
        
        print(f"   ---> {len(reviews)} total reviews")

#Create a dataframe of the reviews
review_df = pd.DataFrame()
review_df['reviews']= reviews
print(review_df)

#Convert the Dataframe to csv
review_df.to_csv('BA_reviews.csv')
print(review_df)

reviews = pd.read_csv('BA_reviews.csv')
reviews = reviews.pop('reviews')
print(reviews)


