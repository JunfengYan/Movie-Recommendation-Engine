# Databricks notebook source
# MAGIC %md 
# MAGIC ### Spark HW2 Moive Recommendation
# MAGIC In this notebook, we will use an Alternating Least Squares (ALS) algorithm with Spark APIs to predict the ratings for the movies in [MovieLens small dataset](https://grouplens.org/datasets/movielens/latest/)

# COMMAND ----------

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
%matplotlib inline

# COMMAND ----------

import os
os.environ["PYSPARK_PYTHON"] = "python3"

# COMMAND ----------

import databricks.koalas as ks

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part1: Data ETL and Data Exploration

# COMMAND ----------

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("moive analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

movies_df = spark.read.load("/FileStore/tables/movies.csv", format='csv', header = True)
ratings_df = spark.read.load("/FileStore/tables/ratings.csv", format='csv', header = True)
links_df = spark.read.load("/FileStore/tables/links.csv", format='csv', header = True)
tags_df = spark.read.load("/FileStore/tables/tags.csv", format='csv', header = True)

# COMMAND ----------

movies_df.show(5)

# COMMAND ----------

ratings_df.show(5)

# COMMAND ----------

links_df.show(5)

# COMMAND ----------

tags_df.show(5)

# COMMAND ----------

tmp1 = ratings_df.groupBy("userID").count().toPandas()['count'].min()
tmp2 = ratings_df.groupBy("movieId").count().toPandas()['count'].min()
print('For the users that rated movies and the movies that were rated:')
print('Minimum number of ratings per user is {}'.format(tmp1))
print('Minimum number of ratings per movie is {}'.format(tmp2))

# COMMAND ----------

tmp1 = sum(ratings_df.groupBy("movieId").count().toPandas()['count'] == 1)
tmp2 = ratings_df.select('movieId').distinct().count()
print('{} out of {} movies are rated by only one user'.format(tmp1, tmp2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Part 1: Spark SQL and OLAP 

# COMMAND ----------

movies_df.registerTempTable("movies")
ratings_df.registerTempTable("ratings")
links_df.registerTempTable("links")
tags_df.registerTempTable("tags")

# COMMAND ----------

display(ratings_df)


# COMMAND ----------

# MAGIC %sql
# MAGIC select userId, count(movieId) as rating_count from ratings group by userId order by rating_count desc

# COMMAND ----------

# MAGIC %md ### Q1: The number of Users

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(distinct UserId) as UserCount from ratings

# COMMAND ----------

# MAGIC %md ### Q2: The number of Movies

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(distinct MovieId) as MovieCount from movies

# COMMAND ----------

# MAGIC %md ### Q3:  How many movies are rated by users? List movies not rated before

# COMMAND ----------

# MAGIC %sql 
# MAGIC Select a.title,a.genres from movies a left join ratings b on a.movieId = b.movieId where b.rating is null

# COMMAND ----------

# MAGIC %sql 
# MAGIC select count(distinct movieId) from ratings

# COMMAND ----------

# MAGIC %md ### Q4: List Movie Genres

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct genres from movies

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct genre from movies
# MAGIC lateral view explode(split(genres,'[|]')) as genre order by genre

# COMMAND ----------

# MAGIC %md ### Q5: Movie for Each Category

# COMMAND ----------

# MAGIC %sql
# MAGIC select distinct Category,count(movieId) as number from movies
# MAGIC lateral view explode(split(genres,'[|]')) as Category group by Category order by number desc

# COMMAND ----------

# MAGIC %sql 
# MAGIC Select t.Category, concat_ws(',',collect_set(t.title)) as list_of_movies from
# MAGIC (select Category,title from movies 
# MAGIC  lateral view explode(split(genres,'[|]')) as Category 
# MAGIC  group by Category,title 
# MAGIC  order by Category) as t
# MAGIC group by t.Category

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Part2: Spark ALS based approach for training model
# MAGIC We will use an Spark ML to predict the ratings, so let's reload "ratings.csv" using ``sc.textFile`` and then convert it to the form of (user, item, rating) tuples.

# COMMAND ----------

ratings_df.show()

# COMMAND ----------

movie_ratings=ratings_df.drop('timestamp')

# COMMAND ----------

# Data type convert
from pyspark.sql.types import IntegerType, FloatType
movie_ratings = movie_ratings.withColumn("userId", movie_ratings["userId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("movieId", movie_ratings["movieId"].cast(IntegerType()))
movie_ratings = movie_ratings.withColumn("rating", movie_ratings["rating"].cast(FloatType()))

# COMMAND ----------

movie_ratings.count()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### ALS Model Selection and Evaluation
# MAGIC 
# MAGIC With the ALS model, we can use a grid search to find the optimal hyperparameters.

# COMMAND ----------

# import package
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder

# COMMAND ----------

#Create test and train set
(training,test)=movie_ratings.randomSplit([0.8,0.2],seed = 2020)

# COMMAND ----------

#Create ALS model
valmodel = ALS(rank = 20,maxIter = 10,regParam = 0.01,userCol = "userId",itemCol = "movieId",ratingCol = 'rating',coldStartStrategy = "drop" )


# COMMAND ----------

#Tune model using ParamGridBuilder
grid = ParamGridBuilder()\
       .addGrid(ALS.rank,[5,20,50])\
       .addGrid(ALS.maxIter,[3,5,10])\
       .addGrid(ALS.regParam,[0.05,0.1,0.15])\
       .build()
        

# COMMAND ----------

# Define evaluator as RMSE
rmse = RegressionEvaluator(predictionCol='prediction',labelCol = 'rating',metricName="rmse")


# COMMAND ----------

# Build Cross validation 
cv = CrossValidator(estimator = valmodel,estimatorParamMaps = grid,evaluator = rmse,numFolds = 5,parall)

# COMMAND ----------

#Fit ALS model to training data
model = valmodel.fit(training)

# COMMAND ----------

#Extract best model from the tuning exercise using ParamGridBuilder
cv = cv.fit(training)

# COMMAND ----------

#choose best model
bestmodel = cv.bestModel


print ("**Best Model**")
print (" Rank:" + str(bestmodel.rank))
print (" MaxIter:" + str(bestmodel._java_obj.parent().getMaxIter()))
print (" RegParam:" + str(bestmodel._java_obj.parent().getRegParam()))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Model testing
# MAGIC And finally, make a prediction and check the testing error.

# COMMAND ----------

#trainning error
predictions = bestmodel.transform(training)
rmse_1 = rmse.evaluate(predictions)
#testing error
predictions=bestmodel.transform(test)
rmse_2 = rmse.evaluate(predictions)

# COMMAND ----------

#Print evaluation metrics and model parameters
#pretty bad overfitting
print ("training RMSE = "+str(rmse_1))
print("testing RMSE="+str(rmse_2))
print ("**Best Model**")
print (" Rank:"), 
print (" MaxIter:"), 
print (" RegParam:"), 

# COMMAND ----------

#better
valmodel = ALS(rank = 20,maxIter = 10,regParam = 0.1,userCol = "userId",itemCol = "movieId",ratingCol = 'rating',coldStartStrategy = "drop" )
model = valmodel.fit(training)
predict = model.transform(training)
predict_2 = model.transform(test)
rmse_3 = rmse.evaluate(predict)
rmse_4 = rmse.evaluate(predict_2)
print(str(rmse_3) + "  " + str(rmse_4) )

# COMMAND ----------


type(model)

# COMMAND ----------

predict_2.show()

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Model apply and see the performance

# COMMAND ----------

alldata=model.transform(movie_ratings)
rmse_2 = rmse.evaluate(alldata)
print ("RMSE = "+str(rmse_2))

# COMMAND ----------

alldata.registerTempTable("alldata")

# COMMAND ----------

# MAGIC %sql select * from alldata

# COMMAND ----------

# MAGIC %sql select * from movies join alldata on movies.movieId=alldata.movieId

# COMMAND ----------

model.extractParamMap()

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Recommend moive to users with id: 575, 232. 
# MAGIC you can choose some users to recommend the moives 

# COMMAND ----------

#recommend top k products for user

topkrecs = model.recommendForAllUsers(10)

# COMMAND ----------

from pyspark.sql import functions as F

# COMMAND ----------

movies_df.show()

# COMMAND ----------

user_list=[575, 232]

user_recs = topkrecs.where(topkrecs.userId.isin(user_list))\
            .select('userId',F.explode('recommendations'))\
            .select('userId',F.col('col').movieId.alias('movieId'),F.col('col').rating.alias('rating'))\
            .join(movies_df,'movieId','left')\
            .select('userId','movieId','title','genres','rating')


# COMMAND ----------

recs = user_recs.to_koalas()

# COMMAND ----------

recs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Find the similar moives for moive with id: 463, 471
# MAGIC You can find the similar moives based on the ALS results

# COMMAND ----------

# MAGIC %md 
# MAGIC 
# MAGIC ## Write the report 
# MAGIC motivation
# MAGIC 1. step1
# MAGIC 2. step2
# MAGIC 3. step3
# MAGIC 4. step4  
# MAGIC output and conclusion