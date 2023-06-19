# Fake News Detection

[Try out the model here](https://fake-news-b8e02d374446.herokuapp.com/)

This is a small Flask webapp built around the model I trained in [this kaggle notebook](https://www.kaggle.com/code/liamgeron/99-9-accurate-news-classification-with-bert). I was able to achieve > 99.9% accuracy with it using BERT, so I decided to put it online so others can play around with it.

NOTE: I actually re-trained this model using TinyBERT to meet Heroku's size specifications, but I was still able to achieve > 99.9% accuracy at around 1/10th of the size.

The Flask app was deployed using Heroku.

If you enjoy it, please support me by starring the repo and/or upvoting the notebook in kaggle!
