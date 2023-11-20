from os import read
from flask import Flask, request, render_template

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

import pickle

popular_50_courses = pickle.load(open('popular_50_courses.pkl', 'rb'))
popular_courses = pickle.load(open('popular_courses.pkl', 'rb'))
courses_crs = pickle.load(open('courses_crs.pkl', 'rb'))
vectors = pickle.load(open('vectors.pkl', 'rb'))


app = Flask(__name__)

def similarity(vectors):
    return cosine_similarity(vectors)


@app.route('/')

def index():
    return render_template('index.html',
                           course_name = list(popular_50_courses['title'].values),
                           course_url = list(popular_50_courses['course_url'].values),
                           )


@app.route('/recommend')
def recommend_ui():
    return render_template('recommend.html')

@app.route('/recommend_courses', methods=['POST'])
def recommend():
    user_input = request.form.get('user_input')
    course_index = courses_crs[courses_crs['title'] == user_input].index[0]
    similarity = cosine_similarity(vectors)
    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)), reverse = True, key = lambda x: x[1])[1:5]
    data = []
    
    for i in course_list:
        item =[]
        temp_df = popular_courses[popular_courses['title'] == popular_courses.iloc[i[0]].title]
        item.extend(list(temp_df['title'].values))
        item.extend(list(temp_df['instructor_name'].values))
        item.extend(list(temp_df['course_url'].values))
        
        data.append(item)
      
    return render_template('recommend.html', data=data)

if __name__=="__main__":
    app.run(debug=True)

