import pandas as pd
import numpy as np


#We are predicting movie ratings
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3))

#Pandas groupby the movies and then find the average rating for each movie
movieProperties = ratings.groupby('movie_id').agg({'rating': [np.size, np.mean]})


movieNumRatings = pd.DataFrame(movieProperties['rating']['size'])

#Normalize the ratings based on viewership on a scale of 0-1
movieNormalizedNumRatings = movieNumRatings.apply(lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)))
movieNormalizedNumRatings.head()


movieDict = {}
with open(r'ml-100k/u.item', encoding="ISO-8859-1") as f:
    temp = ''
    for line in f:
        #line.decode("ISO-8859-1")
        fields = line.rstrip('\n').split('|')
        movieID = int(fields[0])
        name = fields[1]
        genres = fields[5:25]
        genres = map(int, genres)
        movieDict[movieID] = (name, np.array(list(genres)), movieNormalizedNumRatings.loc[movieID].get('size'), movieProperties.loc[movieID].rating.get('mean'))
