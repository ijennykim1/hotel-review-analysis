from bs4 import BeautifulSoup
from collections import Counter
import re
import pandas
import numpy
from matplotlib import pyplot
import os
import nltk
from nltk.corpus import stopwords
import operator
from wordcloud import WordCloud
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats

#Function that opens a file in a different folder
def open_file(directory):
    soup_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".html"):
            with open(directory+"/"+filename) as file_reader:
                soup_list.append(BeautifulSoup(file_reader,"lxml"))
    return soup_list

#Function that creates a dataframe
def create_df(soup_list):
    form = ['Hotel Name','Location','Address','Review','Rating']
    df = pandas.DataFrame(columns=form)
    hotel_name = soup_list[0].find(class_="ui_header h1").get_text()
    street_address = soup_list[0].find(class_="street-address").get_text()
    extended_address = soup_list[0].find(class_="extended-address")
    if extended_address is None:
        extended_address = ""
    else:
        extended_address = extended_address.get_text()
    hotel_city = soup_list[0].find(class_="locality").get_text()
    city_state = re.findall("\D*",hotel_city)[0]
    hotel_address = street_address + " " + extended_address + " " + city_state

    review_tag = []
    for soup in soup_list:
        review_tag+=(soup.find_all(class_="hotels-review-list-parts-SingleReview__mainCol--2XgHm"))

    for review in review_tag:
        comment = review.find("q",class_="hotels-review-list-parts-ExpandableReview__reviewText--3oMkH").get_text()
        rating = review.find(class_="hotels-review-list-parts-RatingLine__bubbles--1oCI4")
        bubble = str(rating.find("span"))
        star = pandas.to_numeric(re.findall("\d",bubble)[0])
        df = df.append({'Hotel Name':hotel_name,'Location':city_state,'Address':hotel_address,'Review':comment,'Rating':star},ignore_index=True)

    return df

nyh = open_file("nyhilton")
nyhilton_df = create_df(nyh)
nyhilton_df["Hotel Code"] = "NYH"

wkh = open_file("waikihilton")
waikihilton_df = create_df(wkh)
waikihilton_df["Hotel Code"] = "WKH"

#Function that returns a Counter of words in a review
pattern = re.compile("\w+")
def find_words(df):
    wc = Counter()
    for comment in df['Review']:
        comment = comment.lower()
        words = pattern.findall(comment)
        wc.update(words)
    return wc

#type=Counter
words_in_nyh = find_words(nyhilton_df)
words_in_wkh = find_words(waikihilton_df)
total_words_in_hotel = words_in_nyh + words_in_nys + words_in_wkh + words_in_wks

#Function that returns a Counter of refined words in a review
stop_words = set(stopwords.words('english'))
def remove_stopwords(words_in_hotel):
    for word in list(words_in_hotel):
        if word in stop_words:
            del words_in_hotel[word]
    return words_in_hotel

no_stop_words_in_nyh = remove_stopwords(words_in_nyh)
no_stop_words_in_wkh = remove_stopwords(words_in_wkh)

#Function that returns the total number of words
def total_words(words_in_hotel):
    tot_val = 0
    for value in words_in_hotel.values():
        tot_val += value
    return tot_val

#Function that calculates log probability of a word
vocab_size = len(total_words_in_hotel)
def log_prob(sorted_unique_hotel,tot_words_hotel,n):
    word = []
    prob = []
    for (key,val) in list(sorted_unique_hotel)[:n]:
        word.append(key)
        count = val + 1
        token_tot = tot_words_hotel + vocab_size
        log_prob = numpy.log(count/token_tot)
        prob.append(log_prob)
    return word, prob
