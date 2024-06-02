import pandas as pd


#We are trying to find relationships between movies which people like to, basically a so and so liked this movie and you both liked the same movie so you may like this one 

#Load the features
r_cols = ['user_id', 'movie_id', 'rating']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=r_cols, usecols=range(3), encoding="ISO-8859-1")


#Load the DV
m_cols = ['movie_id', 'title']
movies = pd.read_csv('ml-100k/u.item', sep='|', names=m_cols, usecols=range(2), encoding="ISO-8859-1")


#Large data frame
ratings = pd.merge(movies, ratings)

ratings.head()

#Creates a data table where the rows are the user_id and the cols are the titles and if there is a rating then the value is the rating
userRatings = ratings.pivot_table(index=['user_id'],columns=['title'],values='rating')
userRatings.head()

#Finds the correlations of the ratings
corrMatrix = userRatings.corr()
corrMatrix.head()

#removes correlations where only a few users rated the pair well together (minimum is 100 atleast)
corrMatrix = userRatings.corr(method='pearson', min_periods=100)
corrMatrix.head()