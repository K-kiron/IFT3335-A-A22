import sys
import numpy as np
import pandas as pd
import csv

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from nltk.stem import LancasterStemmer
from nltk.tokenize import word_tokenize
from openpyxl import Workbook
from nltk.corpus import stopwords
from sklearn.model_selection import cross_validate
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

# 停词
CATEGORIES_STOPWORDS = ['/.', '/,', '[', ']', '/(', '/)', '\n', "/'", "/''", '/``', '/:']
CATEGORIES_STOPWORDS_EXTRA = ['/IN', '/DT', '/CC']

# 去停词代码
STOPLIST_TXT = pd.read_csv('stoplist.txt', header=None)[0].values.tolist()
PUNCTUATION_STOPWORDS = ['======================================', '.', ',', '[', ']', '(', ')', '\n', "'", "''", '``',
                         ':', ';', '{', '}', '"'] + STOPLIST_TXT
NG_STOPWORDS = ['======================================', '.', ',', '(', ')', '\n', "'", "''", '``',
                ':', ';', '{', '}', '"'] + STOPLIST_TXT

# 不去停词
# PUNCTUATION_STOPWORDS = ['.', ',', '[', ']', '(', ')', '\n', "'", "''", '``', ':', ';', '{', '}', '"']


# Config
DATASETS = ['gc', 'nw']
MODELS = ['nb', 'dt', 'mlp']
# CONFIG = {
#     'stopwords': 'all',  # ['all', 'punctuation', 'none']
#     'gc': {
#         'filename': 'gc_dataset.csv',
#         'n_words': 2,
#         'vectorizer': 'count',  # ['count', 'tfidf']
#         'mlp': {
#             'solver': 'sgd',  # ['lbfgs', 'sgd', 'adam']
#             'hidden_layer_sizes': (40)
#         },
#         'dt': {
#             'depth': 30
#         }
#     }
# }
CONFIG = {
    'stopwords': 'all',  # ['all', 'punctuation', 'none']
    'gc': {
        'filename': 'gc_dataset.csv',
        'n_words': 2,
        'vectorizer': 'tfidf',  # ['count', 'tfidf']
        'mlp': {
            'solver': 'adam',  # ['lbfgs', 'sgd', 'adam']
            'hidden_layer_sizes': (50, 100)
            , 'activation': 'tanh'  # ['relu','logistic','tanh']
        },
        'dt': {
            'depth': 50
        }},
    'nw': {
        'filename': 'nw_dataset.csv',
        'n_words': 2,
        'vectorizer': 'count',  # ['count', 'tfidf']
        'mlp': {
            'solver': 'adam',  # ['lbfgs', 'sgd', 'adam']
            'hidden_layer_sizes': (50, 100),
            'activation': 'tanh'  # ['relu','logistic','tanh']
        },
        'dt': {
            'depth': 500
        }},
    'ng': {
        'filename': 'ng_dataset.csv',
        'n_words': 2,
        'vectorizer': 'count',  # ['count', 'tfidf']
        'mlp': {
            'solver': 'adam',  # ['lbfgs', 'sgd', 'adam']
            'hidden_layer_sizes': (50, 100),
            'activation': 'tanh'  # ['relu','logistic','tanh']
        },
        'dt': {
            'depth': 500
        }}
}


# Grammatical classification extraction
def gc_extraction(stopwords=False, extra_stopwords=False, custom_n_words=None):
    if custom_n_words is None:
        n_words = CONFIG['gc']['n_words']
    else:
        n_words = custom_n_words
    data = []
    with open('corpus.txt') as file:
        lines = file.readlines()
    separator = lines[1]
    lines.remove(separator)
    for line in lines:
        tmp = []
        line = line.split(' ')
        try:
            line.remove('======================================')
        except:
            pass
        if stopwords:
            for i in range(len(line)):
                stopword_found = False
                for stopword in CATEGORIES_STOPWORDS:
                    if line[i].find(stopword) != -1:
                        stopword_found = True
                        break
                if not stopword_found and extra_stopwords:
                    for stopword in CATEGORIES_STOPWORDS_EXTRA:
                        if line[i].find(stopword) != -1:
                            stopword_found = True
                            break
                if not stopword_found:
                    for stopword in STOPLIST_TXT:
                        if line[i].split("/")[0] == stopword:
                            stopword_found = True
                            break
                if not stopword_found:
                    tmp.append(line[i])
            line = tmp
        nulls = []
        for _ in range(n_words):
            nulls.append('/VOID')
        line = nulls + line + nulls
        target_word_found = False
        category = ''
        for i in range(len(line)):
            if line[i].find('interest_') == 0:
                # 取interest的类型数字
                category = line[i][9:10]
                line = line[i - n_words:i + n_words + 1]
                target_word_found = True
                break
            elif line[i].find('interests_') == 0:
                category = line[i][10:11]
                line = line[i - n_words:i + n_words + 1]
                line.pop(n_words)
                target_word_found = True
                break
        if target_word_found:
            line.pop(n_words)
            for i in range(len(line)):
                try:
                    line[i] = line[i].split('/')[1]
                except:
                    line[i] = 'VOID'
            line = ' '.join(line)
            line = line.replace('VOID', '')
            line = [category, line]
            data.append(line)
    header = ['label', 'features']
    with open(CONFIG['gc']['filename'], 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
    print(CONFIG['gc']['filename'] + ' generated')


# n-words feature extract
def nw_extraction(punctuation_stopwords=True, custom_n_words=None):
    lancaster_stemmer = LancasterStemmer()
    if custom_n_words is None:
        n_words = CONFIG['nw']['n_words']
    else:
        n_words = custom_n_words
    data = []
    with open('corpus.txt') as file:
        lines = file.readlines()
    separator = lines[1]
    lines.remove(separator)
    for line in lines:
        line = word_tokenize(line)
        try:
            line.remove('======================================')
        except:
            pass
        for i in range(len(line)):
            try:
                line[i] = line[i].split('/')[0]
            except:
                line[i] = 'VOID'
        for i in range(len(line)):
            line[i] = lancaster_stemmer.stem(line[i])
        if punctuation_stopwords:
            line = list(filter(lambda x: x not in PUNCTUATION_STOPWORDS, line))
        nulls = []
        for _ in range(n_words):
            nulls.append('VOID')
        line = nulls + line + nulls
        target_word_found = False
        category = ''
        for i in range(len(line)):
            if line[i].find('interest_') == 0:
                category = 'C' + line[i][9:10]
                line = line[i - n_words:i + n_words + 1]
                # 删除n_words+1的元素，即interest
                line.pop(n_words)
                target_word_found = True
                break
            elif line[i].find('interests_') == 0:
                category = 'C' + line[i][10:11]
                line = line[i - n_words:i + n_words + 1]
                line.pop(n_words)
                target_word_found = True
                break
            elif not line[i].isalpha():
                line[i] = 'NUM'
        if target_word_found:
            line = ' '.join(line)
            line = line.replace('VOID', '')
            line = [category, line]
            data.append(line)
    with open(CONFIG['nw']['filename'], 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(CONFIG['nw']['filename'] + ' generated')


# nominal group extract
def ng_extraction(punctuation_stopwords=True):
    lancaster_stemmer = LancasterStemmer()
    data = []
    with open('corpus.txt') as file:
        lines = file.readlines()
    separator = lines[1]
    lines.remove(separator)
    for line in lines:
        line = word_tokenize(line)
        try:
            line.remove('======================================')
        except:
            pass
        for i in range(len(line)):
            try:
                line[i] = line[i].split('/')[0]
            except:
                line[i] = 'VOID'
        for i in range(len(line)):
            line[i] = lancaster_stemmer.stem(line[i])
        if punctuation_stopwords:
            line = list(filter(lambda x: x not in NG_STOPWORDS, line))
        target_word_found = False
        temp = []
        inside_brackets = False
        category = ''
        for word in line:
            if word == '[':
                inside_brackets = True
            elif word == ']':
                inside_brackets = False
                if not target_word_found:
                    temp = []
            elif inside_brackets:
                if word.find('interest_') == 0:
                    category = 'C' + word[9:10]
                    target_word_found = True
                elif word.find('interests_') == 0:
                    category = 'C' + word[10:11]
                    target_word_found = True
                elif not word.isalpha():
                    word = 'NUM'
                temp.append(word)
        if target_word_found:
            for i in range(len(temp)):
                if temp[i].find('interest_') == 0:
                    temp[i] = temp[i][10:]
                elif temp[i].find('interests_') == 0:
                    temp[i] = temp[i][11:]
            temp = ' '.join(temp)
            temp = temp.replace('VOID', '')
            if temp == '':
                temp = 'VOID'
            temp = [category, temp]
            data.append(temp)
    with open(CONFIG['ng']['filename'], 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    print(CONFIG['ng']['filename'] + ' generated')


gc_extraction()
nw_extraction()
ng_extraction()
# whole_sentence_extraction()

# datasets = ['gc']
# datasets = ['gc', 'nw']
# datasets = ['nw']
datasets = ['gc', 'nw', 'ng']
# datasets = ['ng']
models = ['naive_bayes', 'decision_tree', 'random_forest', 'svm', 'mlp']


# models =  ['mlp']

# Draw the learing curve for each algorithm, show the comparison in one pic, save as png into the folder 'img'
def draw_learning_curve(dataset, classifier):
    # load data
    df = pd.read_csv(CONFIG[dataset]['filename'], header=None)
    X = df.iloc[:, 1]
    y = df.iloc[:, 0]
    # split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    # vectorize the text
    vectorizer = TfidfVectorizer()
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
    # draw the learning curve
    train_sizes, train_scores, test_scores = learning_curve(classifier, X_train, y_train, cv=5, n_jobs=-1,
                                                            train_sizes=np.linspace(.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.title('Learning Curve for ' + CONFIG[dataset]['filename'] + ' dataset')
    plt.xlabel('Training examples')
    plt.ylabel('Score')
    plt.savefig('img/' + CONFIG[dataset]['filename'] + '_' + classifier.__class__.__name__ + '.png')
    plt.clf()








for current_dataset in datasets:
    print(CONFIG[current_dataset])
    data = pd.read_csv(CONFIG[current_dataset]['filename'], header=None)
    print('Dataset: ' + current_dataset)
    y = data[0].values
    X = data[1]
    if CONFIG[current_dataset]['vectorizer'] == 'count':
        vectorizer = CountVectorizer(ngram_range=(1, 2))
    else:
        vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    for current_model in models:
        if current_model == 'naive_bayes':
            nb = MultinomialNB()
            nb.fit(X_train, y_train)
            y_pred = nb.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            print('naive_bayes Accuracy: ' + str('{:.4f}'.format(acc)))
            draw_learning_curve(current_dataset, nb)
        elif current_model == 'decision_tree':
            dt = DecisionTreeClassifier(max_depth=CONFIG[current_dataset]['dt']['depth'])
            dt.fit(X_train, y_train)
            y_pred = dt.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            print('decision_tree Accuracy: ' + str('{:.4f}'.format(acc)))
            draw_learning_curve(current_dataset, dt)
        elif current_model == 'random_forest':
            clf = RandomForestClassifier(max_depth=12, random_state=42)
            clf = clf.fit(X_train, y_train)
            y_test = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            print('random_forest Accuracy: ' + str('{:.4f}'.format(acc)))
            draw_learning_curve(current_dataset, clf)

        elif current_model == 'svm':
            kernels = ['sigmoid', 'rbf']
            clf = svm.SVC(kernel=kernels[1]).fit(X_train, y_train)
            y_test = clf.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            print('svm Accuracy: ' + str('{:.4f}'.format(acc)))
            draw_learning_curve(current_dataset, clf)

        elif current_model == 'mlp':
            mlp = MLPClassifier(hidden_layer_sizes=CONFIG[current_dataset]['mlp']['hidden_layer_sizes']
                                , max_iter=3000
                                , solver=CONFIG[current_dataset]['mlp']['solver']
                                , activation=CONFIG[current_dataset]['mlp']['activation'])
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            print('mlp Accuracy: ' + str('{:.4f}'.format(acc)))
            draw_learning_curve(current_dataset, mlp)
