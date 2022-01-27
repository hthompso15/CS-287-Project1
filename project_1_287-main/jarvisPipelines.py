# Jarvis Pipelines
# for data white paper

# imports
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC

from sklearn.model_selection import cross_val_score, ShuffleSplit, learning_curve

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

# helper functions
def count_unique_words(text):
    return len(set(text.split()))

def count_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)

def run_learning_cruve(pipe, X, y, cv=None, n_jobs=1, train_sizes=np.linspace(.1, 1.0, 10)):
  train_sizes, train_scores, test_scores = learning_curve(
        pipe, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
  train_mean = np.mean(train_scores, axis=1)
  train_std = np.std(train_scores, axis=1)
  test_mean = np.mean(test_scores, axis=1)
  test_std = np.std(test_scores, axis=1)
  return train_mean, train_std, test_mean, test_std

# read data from jarvis
file = 'jarvis.csv'
df = pd.read_csv(file)
df.sample(5)

# check Nan
df.isna().sum()

# feature engineering
analyzer = SentimentIntensityAnalyzer()
df['character_length'] = df['message'].str.len()
df['word_count'] = df['message'].str.split().str.len()
df['characters_per_word'] = df['character_length']/df['word_count']
df['unique_word_count'] = df["message"].apply(lambda x:count_unique_words(x))
df['stopword_count'] = df["message"].apply(lambda x:count_stopwords(x))
df['unique_vs_words'] = df['unique_word_count']/df['word_count']
df['stopwords_vs_words'] = df['stopword_count']/df['word_count']
df['question_mark'] = df['message'].apply(lambda x: len([x for x in x.split() if x.endswith('?')]))
df['period'] = df['message'].apply(lambda x: len([x for x in x.split() if x.endswith('.')]))
df['compound'] = [analyzer.polarity_scores(x)['compound'] for x in df['message']]
df['neg'] = [analyzer.polarity_scores(x)['neg'] for x in df['message']]
df['neu'] = [analyzer.polarity_scores(x)['neu'] for x in df['message']]
df['pos'] = [analyzer.polarity_scores(x)['pos'] for x in df['message']]

# get X y cols
X, y = df['message'], df['action']
target_names = set(y.tolist())


# split data
internal = df.iloc[0:64]
external = df.iloc[64:]
print(internal.head())
print(external.head())

# pipelines used in data white paper
nb = Pipeline([
  ('vect', CountVectorizer()),
  ('tfidf', TfidfTransformer()),
  ('clf', MultinomialNB()),
        ])

svc = Pipeline([
  ('vect', CountVectorizer()),
  ('tfidf', TfidfTransformer()),
  ('clf', SVC())
  ])


testing_size = [0.1,0.2,0.3,0.4, 0.5, 0.6,0.7,0.8,0.9]

# data white paper question 3

# run cross_val_score on both nb and svc
mean_lst=[]; std_lst=[]
for i in range(len(testing_size)):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_size[i], random_state=42)
  score = cross_val_score(nb, X_train, y_train, cv=5, scoring='accuracy')
  mean = score.mean()
  std = score.std()*2
  print("%s val accuracy: %0.3f (+/- %0.3f)" % (testing_size[i], mean, std * 2))
  mean_lst.append(mean); std_lst.append(std)

mean_lst2=[]; std_lst2=[]
for i in range(len(testing_size)):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testing_size[i], random_state=42)
  score = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy')
  mean = score.mean()
  std = score.std()*2
  print("%s val accuracy: %0.3f (+/- %0.3f)" % (testing_size[i], mean, std * 2))
  mean_lst2.append(mean); std_lst2.append(std)

# plotting
fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=100, sharex=False, sharey=True)
axes = axes.ravel()

axes[0].plot([i * 100 for i in testing_size], mean_lst, linestyle='-', color='darkorange', linewidth=2.0)
axes[0].fill_between(
    [i * 100 for i in testing_size],
    [a - b for a, b in zip(mean_lst, std_lst)],
    [a + b for a, b in zip(mean_lst, std_lst)],
    alpha=0.2,
    color="darkorange",
    lw=2
)
axes[0].set_title('MultinomialNB')
axes[0].set_ylabel('Cross validation score')
axes[0].set_xlabel('% of data removed')

axes[1].plot([i * 100 for i in testing_size], mean_lst2, linestyle='-', color='navy', linewidth=2.0)
axes[1].fill_between(
    [i * 100 for i in testing_size],
    [a - b for a, b in zip(mean_lst2, std_lst2)],
    [a + b for a, b in zip(mean_lst2, std_lst2)],
    alpha=0.2,
    color="navy",
    lw=2
)
axes[1].set_title('SVM')
axes[1].set_xlabel('% of data removed')

fig.tight_layout()
fig.show()

## create new X and y for external and internal data
Xe, ye = external.message, external.action
Xi, yi = internal.message, internal.action


# new settings for question 4
train_sizes = np.linspace(.1, 1.0, 10)
cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)


# plot question 4
fig, axes = plt.subplots(3, 2, figsize=(8, 10.5), dpi=100, sharex=False, sharey=True)
axes = axes.ravel()
plt.xlabel("Training examples")

train_mean, train_std, test_mean, test_std = run_learning_cruve(nb, Xi, yi, cv=cv, n_jobs=4)
axes[0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="darkorange")
axes[0].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="navy")
axes[0].plot(train_sizes, train_mean, 'o-', color="darkorange", label="Training score")
axes[0].plot(train_sizes, test_mean, 'o-', color="navy", label="Cross-validation score")
axes[0].set_title('Naive Bayes - Internal')
axes[0].yaxis.grid(True)

train_mean, train_std, test_mean, test_std = run_learning_cruve(svc, Xi, yi, cv=cv, n_jobs=4)
axes[1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="darkorange")
axes[1].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="navy")
axes[1].plot(train_sizes, train_mean, 'o-', color="darkorange", label="Training score")
axes[1].plot(train_sizes, test_mean, 'o-', color="navy", label="Cross-validation score")
axes[1].set_title('SVM - Internal')
axes[1].yaxis.grid(True)

train_mean, train_std, test_mean, test_std = run_learning_cruve(nb, Xe, ye, cv=cv, n_jobs=4)
axes[2].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="darkorange")
axes[2].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="navy")
axes[2].plot(train_sizes, train_mean, 'o-', color="darkorange", label="Training score")
axes[2].plot(train_sizes, test_mean, 'o-', color="navy", label="Cross-validation score")
axes[2].set_title('Naive Bayes - External')
axes[2].yaxis.grid(True)

train_mean, train_std, test_mean, test_std = run_learning_cruve(svc, Xe, ye, cv=cv, n_jobs=4)
axes[3].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="darkorange")
axes[3].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="navy")
axes[3].plot(train_sizes, train_mean, 'o-', color="darkorange", label="Training score")
axes[3].plot(train_sizes, test_mean, 'o-', color="navy", label="Cross-validation score")
axes[3].set_title('SVM - External')
axes[3].yaxis.grid(True)

train_mean, train_std, test_mean, test_std = run_learning_cruve(nb, X, y, cv=cv, n_jobs=4)
axes[4].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="darkorange")
axes[4].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="navy")
axes[4].plot(train_sizes, train_mean, 'o-', color="darkorange", label="Training score")
axes[4].plot(train_sizes, test_mean, 'o-', color="navy", label="Cross-validation score")
axes[4].set_title('Naive Bayes - All')
axes[4].yaxis.grid(True)
axes[4].set_xlabel('Training examples')

train_mean, train_std, test_mean, test_std = run_learning_cruve(svc, X, y, cv=cv, n_jobs=4)
axes[5].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="darkorange")
axes[5].fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="navy")
axes[5].plot(train_sizes, train_mean, 'o-', color="darkorange", label="Training score")
axes[5].plot(train_sizes, test_mean, 'o-', color="navy", label="Cross-validation score")
axes[5].set_title('SVM - All')
axes[5].yaxis.grid(True)
axes[5].set_xlabel('Training examples')

#fig.tight_layout()
handles, labels = axes[5].get_legend_handles_labels()
plt.legend(handles=handles, labels=labels,  loc = (-0.8, -0.35), ncol=2)
fig.show()