import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
import string
import re
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from collections import Counter
nltk.download('stopwords')
stemmer = nltk.SnowballStemmer('english')

data = pd.read_csv('BA_reviews.csv',index_col='Unnamed: 0')
print(data.info())
print(data.head())


stopword=set(stopwords.words('english'))
def clean(text):
  text = str(text).lower()
  text = re.sub('\[.*?\]', '', text)
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\n', '', text)
  text = re.sub('\w*\d\w*', '', text)
  text = [word for word in text.split(' ') if word not in stopword]
  text=" ".join(text)
  text = [stemmer.stem(word) for word in text.split(' ')]
  text=" ".join(text)
  text = re.sub('✅ Trip Verified |', '', text)
  text = re.sub('✅', '', text)
  text = re.sub('Trip Verified', '', text)
  text = re.sub('Verified', '', text)
  text = re.sub(' trip verifi', '', text)
  return text
data['reviews'] = data['reviews'].apply(clean)

print(data.head())
print(data.shape)


freq_words = pd.Series(" ".join(data['reviews']).lower().split()).value_counts()[:50]
print(freq_words)

#To plot the frequent words
plt.figure(figsize=(10,10))
freq_words.plot.barh(x=freq_words[0], y=freq_words[1])
plt.show()

#An analysis of the kind of words used in the review
text = " ".join(i for i in data.reviews)
stopwords = set(STOPWORDS)
wordcloud =WordCloud(stopwords=stopwords,background_color="white").generate(text)
plt.figure(figsize=(10,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()

#Transform the reviews into positive,neutral and negative
nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
data['Positive'] = [sentiments.polarity_scores(i)["pos"] for i in data['reviews']]
data['Negative'] = [sentiments.polarity_scores(i)['neg'] for i in data['reviews']]
data['Neutral'] = [sentiments.polarity_scores(i)['neu'] for i in data['reviews']]
data = data[['reviews','Positive','Negative','Neutral']]
sum = data[['Positive','Negative','Neutral']].agg(sum)
print(sum)
print(data.head())
sns.countplot(x = sum, hue = sum)
plt.show()


positive =' '.join([i for i in data['reviews'][data['Positive'] > data["Negative"]]])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(positive)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

negative =' '.join([i for i in data['reviews'][data['Positive'] < data["Negative"]]])
stopwords = set(STOPWORDS)
wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(negative)
plt.figure( figsize=(15,10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


