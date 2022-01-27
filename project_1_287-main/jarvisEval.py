# Jarvis evaluation
# for jarvis brain white paper

######## A. Imports ##########################################################

# new to project 02
import nltk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scikitplot as skplt

# word vectors
import spacy
spacy.load("en_core_web_sm")

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score, cross_val_predict
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD

# Experimentation
from sklearn.feature_selection import chi2, SelectKBest

# sparseInteractions
from scipy import sparse
from itertools import combinations

# nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

################ B. Functions #############################################

def generate_numeric_features(lst):
    analyzer = SentimentIntensityAnalyzer()

    character_length = [len(text) for text in lst]
    word_count = [len(text.split()) for text in lst]
    characters_per_word = [i / j for i, j in zip(character_length, word_count)]
    unique_word_count = [len(set(text.split())) for text in lst]
    stopword_count = [count_stopwords(text) for text in lst]
    unique_vs_words = [i / j for i, j in zip(unique_word_count, word_count)]
    stopwords_vs_words = [i / j for i, j in zip(stopword_count, word_count)]
    #compound = [analyzer.polarity_scores(text)['compound'] for text in lst]
    neg = [analyzer.polarity_scores(text)['neg'] for text in lst]
    neu = [analyzer.polarity_scores(text)['neu'] for text in lst]
    pos = [analyzer.polarity_scores(text)['pos'] for text in lst]

    nparray = np.array(
        [character_length, word_count, characters_per_word, unique_word_count, stopword_count, unique_vs_words,
         stopwords_vs_words, neg, neu, pos])
    return np.transpose(nparray)

def count_stopwords(text):
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    stopwords_x = [w for w in word_tokens if w in stop_words]
    return len(stopwords_x)

######## C. Main ##########################################################

df = pd.read_csv('jarvis.csv')
X, y = df['message'], df['action']
target_names = set(y.tolist())

#  test_size size sets to 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# SVD
tfidf = TfidfVectorizer()
matrix = tfidf.fit_transform(X)
feature_names = tfidf.get_feature_names()
svd = TruncatedSVD(random_state=1)
svd.fit(matrix)
print(svd.singular_values_)
best_fearures = [feature_names[i] for i in svd.components_[0].argsort()[::-1]]
best_fearures[:20]


# SVD plot
ax = skplt.decomposition.plot_pca_2d_projection(svd, matrix, y,
                                           figsize=(10,10),
                                           title='SVD Explained Variances')
ax.set_xlabel = 'First SVD factor'
ax.set_ylabel = 'Second SVD factor'
ax.figure.savefig('SVD.png')

# models
nb = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB())
])

sgdc = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='log', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None))
])

lr = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', LogisticRegression(random_state=42))
])

svm = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(random_state=42, probability=True))
])

dt = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', DecisionTreeClassifier(random_state=42))
])

ab = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', AdaBoostClassifier())])

gb = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', GradientBoostingClassifier())
])

knn = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', KNeighborsClassifier(n_neighbors=3))
])

# model pipeline
pipelines = [nb, sgdc, lr, svm, dt, ab, gb, knn]
pipe_dict = {0: 'Naive Bayes', 1: 'Stochastic Gradient Descent', 2: 'Logistic Regression',
              3: 'Support Vector Machine', 4: 'Decision tree', 5: 'AdaBoost Classifier',
              6: 'Gradient Boosting', 7: 'K-Nearest Neighbors'}

######## E. cross validation ##########################################################

# peform 10-fold validation and print the validation score and store the confusion matrix
confusion_matrices = []

cv = KFold(n_splits=10)
for i, pipe in enumerate(pipelines):
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
    print("%s val accuracy: %0.3f (+/- %0.3f)" % (pipe_dict[i], scores.mean(), scores.std() * 2))
    y_pred = cross_val_predict(pipe, X_train, y_train, cv=cv)
    print(metrics.classification_report(y_train, y_pred,
                                        target_names = target_names))
    confusion_matrices.append(metrics.confusion_matrix(y_train, y_pred))


# Confusion Matrices plots:
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(4, 2, figsize=(12, 12), dpi=200, sharex=True, sharey=True)
axes = axes.ravel()

titles = ['Naive Bayes','Stochastic Gradient Descent','Logistic Regression','SVM','Decision Tree','Adaboost','Gradient Boosting','K-Nearest Neighbor']

for i in range(0,8):
  disp = ConfusionMatrixDisplay(
          confusion_matrix = confusion_matrices[i],
          display_labels=target_names,
  )
  disp.plot(ax=axes[i],xticks_rotation=45)
  disp.ax_.set_title(titles[i])
  if (i%2) ==1 :
      disp.ax_.set_ylabel('')
      disp.ax_.set_xlabel('')

  if (i%2) ==0 :
      disp.im_.colorbar.remove()
      #disp.ax_.set_ylabel('')
      disp.ax_.set_xlabel('')
  if i == 6 :
      disp.ax_.set_xlabel('Predicted label')

plt.xlabel("Predicted label")
plt.tight_layout()
plt.show()
plt.savefig('matrix.png')


# ROC plots for micro and macro averages
fig, axes = plt.subplots(4, 2, figsize=(8, 10.5), dpi=200, sharex=True, sharey=True)
axes = axes.ravel()

for i, pipe in enumerate(pipelines):
    pipe.fit(X_train, y_train)
    y_probas = pipe.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y_probas, plot_micro =True, plot_macro=True, ax=axes[i], title=pipe_dict[i], title_fontsize='small', text_fontsize='small', classes_to_plot=[0, 'cold'])
fig.savefig('ROC.png')

# Roc plots for all classes
fig, axes = plt.subplots(3, 1, figsize=(8, 10.5), dpi=200, sharex=True, sharey=True)
axes = axes.ravel()

pipelines = [nb, sgdc, svm]
pipe_dict = {0: 'Naive Bayes', 1: 'Stochastic Gradient Descent', 2: 'Support Vector Machine'}
for i, pipe in enumerate(pipelines):
    pipe.fit(X_train, y_train)
    y_probas = pipe.predict_proba(X_test)
    skplt.metrics.plot_roc(y_test, y_probas, plot_micro =False, plot_macro=False, title=pipe_dict[i], ax=axes[i])
fig.savefig('ROC2.png')

######## D. Further Experiment Ideas  ##########################################################

# # customized class for using word vectors in skylearn
# # The code is taken from https://lvngd.com/blog/spacy-word-vectors-as-features-in-scikit-learn/
class WordVectorTransformer(TransformerMixin,BaseEstimator):
    def __init__(self, model="en_core_web_sm"):
        self.model = model

    def fit(self,X,y=None):
        return self

    def transform(self,X):
        nlp = spacy.load(self.model)
        return np.concatenate([nlp(doc).vector.reshape(1,-1) for doc in X])

# bi-gram
classifier1 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SVC(random_state=42, probability=True))
])

# feature engineering and union / selection
classifier2 = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
        ('union', FeatureUnion(
            transformer_list=[
                ('numeric_features', Pipeline([
                    ('vect', FunctionTransformer(generate_numeric_features, validate=False)),

                ])),
                ('text_features', Pipeline([
                    ('vect', CountVectorizer(tokenizer=word_tokenize, ngram_range=(1, 2))),
                    ('tfidf', TfidfTransformer())
                ]))
            ]
        )),
    ('scale', MaxAbsScaler()),
    #('dim_red', SelectKBest(chi2, 1000)),
    ('clf', SVC(random_state=42))
    ])

# word vectors
classifier3 = Pipeline([
    ('word2vec', WordVectorTransformer()),
    ('clf', SVC(random_state=42))
])