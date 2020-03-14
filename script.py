import numpy as np, pandas as pd
import sklearn.cluster
import pickle 
from flask import Flask, render_template, request, jsonify
import os, random
seed = 123
os.environ['PYTHONHASHSEED']=str(seed)
random.seed(seed)
np.random.seed(seed)

app=Flask(__name__)
@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/input')
def input():
    return render_template('input.html')

def getRecommendations(prediction_input):
    new_cluster_obj = pickle.load(open('./pickled_objs/cluster_obj.pkl', 'rb'))
    new_cluster_movie_rankings = pickle.load(open('./pickled_objs/cluster_movie_rankings.pkl', 'rb'))
    new_movies = pickle.load(open('./pickled_objs/movies.pkl', 'rb'))
    new_ratings = pickle.load(open('./pickled_objs/ratings.pkl', 'rb'))

    prediction_input.fillna(-1, inplace=True)
    new_clusters = new_cluster_obj.fit_predict(prediction_input)
    print(new_clusters[-1])
    movie_mean = pd.DataFrame(new_ratings.groupby('movieId')['rating'].mean())
    new_movies = pd.merge(new_movies, movie_mean, how='right', on='movieId')
    movies_to_recommend = new_cluster_movie_rankings[new_clusters[-1]]
    movies_to_recommend = movies_to_recommend[movies_to_recommend['weighted_score'] > .75]
    movies_to_recommend = movies_to_recommend[movies_to_recommend['count'] > 0.2]
    

    movies_to_recommend = pd.merge(movies_to_recommend, new_movies, on='movieId', how='left')
    idx = (new_ratings['movieId'].value_counts().head(10).index)
    idx = idx.append(new_ratings['movieId'].value_counts()[100:110].index)
    idx = idx.append(new_ratings['movieId'].value_counts()[500:510].index)
    movies_to_recommend = movies_to_recommend[~movies_to_recommend['movieId'].isin(idx)]
    movies_to_recommend = movies_to_recommend.sort_values(['count','weighted_score','mean'], ascending=False)
    return (new_clusters[-1], movies_to_recommend[['title', 'genres']])

@app.route('/result',methods = ['POST'])
def result():
    new_movies = pickle.load(open('./pickled_objs/movies.pkl', 'rb'))
    ratings = pickle.load(open('./pickled_objs/ratings.pkl', 'rb'))
    idx = (ratings['movieId'].value_counts().head(25)).index
    films = ratings[ratings['movieId'].isin(idx)]
    small_movies = new_movies[new_movies['movieId'].isin(films['movieId'])]
    small_ratings = ratings[ratings['movieId'].isin(small_movies['movieId'])]
    ratings_pivot = small_ratings.pivot(index="userId", columns='movieId', values='rating').fillna(-1)

    if request.method == 'POST':
        data_dict = request.form.to_dict()
        df = pd.DataFrame([data_dict], columns=data_dict.keys())
        df.columns = (df.columns).astype(int)
        df = df.replace('', -1)
        copy = ratings_pivot.append(df).fillna(0)
        cluster, result = getRecommendations(copy)
        return render_template("result.html", data=result.to_html(), cluster_group=cluster)

if __name__ == "__main__":
    app.run(debug=True)