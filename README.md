# nlp-movie-reviews-analysis
Rus: Цель проведённого исследования заключалась в многоуровневом анализе корпуса англоязычных кинорецензий с платформы Rotten Tomatoes, с разделением материалов на положительные («Fresh») и отрицательные («Rotten») оценки. _**Выполнен на языке программирования R.**_

**1. Project Overview**:
This project analyzes movie reviews from Rotten Tomatoes to understand how textual features relate to review sentiment (Fresh vs Rotten).

**2. Business Problem**:
Can we automatically understand how critics evaluate movies based on text data, and identify key drivers of positive and negative reviews?

**3. Dataset**:
Public dataset from Kaggle (Rotten Tomatoes reviews)
~10,000 reviews used for analysis

**4. Methods**:
Text preprocessing (cleaning, tokenization, stopwords removal)
TF-IDF analysis
N-gram analysis (unigrams, bigrams)
Topic modeling (LDA, 6 topics)
Sentiment analysis (AFINN + Bing)
Binary classification model

**5. Key Findings**:
Sentiment is not a strong standalone predictor of review category
Critics often use mixed language, even in positive reviews
Topic modeling revealed key dimensions:
storytelling & эмоции
social/political context
directing & production quality
Classification model performed poorly due to:
class imbalance
complexity of language (irony, nuance)

**Limitations & Improvements**:
class imbalance
lack of contextual models
need for BERT / embeddings

**7. Why it matters**
This analysis can be useful for:

+ automated review monitoring
+ sentiment tracking in media
+ recommendation systems
+ marketing insights
