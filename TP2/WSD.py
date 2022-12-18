from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

# stopword
CATEGORIES_STOPWORDS = ['/.', '/,', '[', ']', '/(', '/)', '\n', "/'", "/''", '/``', '/:']
CATEGORIES_STOPWORDS_EXTRA = ['/IN', '/DT', '/CC']

# code stopword
# STOPLIST_TXT = pd.read_csv('tool word list (stoplist).txt',header=None)[0].values.tolist()
# PUNCTUATION_STOPWORDS = ['.', ',', '[', ']', '(', ')', '\n', "'", "''", '``', ':', ';', '{', '}', '"']+STOPLIST_TXT

# no-stopword
PUNCTUATION_STOPWORDS = ['.', ',', '[', ']', '(', ')', '\n', "'", "''", '``', ':', ';', '{', '}', '"']



text_data = open('interest.acl94.txt', 'r').read()
text_data = text_data.split('$$')

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(text_data, text_data, test_size=0.2)


# Transform the text data into a numerical feature representation using the CountVectorizer
count_vectorizer = CountVectorizer()
X_train_count = count_vectorizer.fit_transform(text_data)
# print(X_train_count)
X_test_count = count_vectorizer.transform(X_test)
# print(count_vectorizer.get_feature_names_out())
# print(X_test_count)

# d=pd.DataFrame(X_test_count.toarray(), columns=count_vectorizer.get_feature_names_out())
# print(d)

# Transform the text data into a numerical feature representation using the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(text_data)
# print(X_train_tfidf)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
# print(X_test_tfidf)

# p=pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
# print(p)



# 计算其他词的TF-IDF以及频数
wsd_dict = {}
for file in os.listdir('.'):
    if wsd_word in file:
        wsd_dict[file.replace('.txt', '')] = read_file(file)
 
# 统计每个词语在语料中出现的次数
tf_dict = {}
for meaning, sents in wsd_dict.items():
    tf_dict[meaning] = []
    for word in sent_cut:
        word_count = 0
        for sent in sents:
            example = list(jieba.cut(sent, cut_all=False))
            word_count += example.count(word)
 
        if word_count:
            tf_dict[meaning].append((word, word_count))
 
idf_dict = {}
for word in sent_cut:
    document_count = 0
    for meaning, sents in wsd_dict.items():
        for sent in sents:
            if word in sent:
                document_count += 1
 
    idf_dict[word] = document_count
 
# 输出值
total_document = 0
for meaning, sents in wsd_dict.items():
    total_document += len(sents)
 
# 计算tf_idf值
mean_tf_idf = []
for k, v in tf_dict.items():
    print(k + ':')
    tf_idf_sum = 0
    for item in v:
        word = item[0]
        tf = item[1]
        tf_idf = item[1] * log2(total_document / (1 + idf_dict[word]))
        tf_idf_sum += tf_idf
        print('%s, 频数为: %s, TF-IDF值为: %s' % (word, tf, tf_idf))
 
    mean_tf_idf.append((k, tf_idf_sum))
 
sort_array = sorted(mean_tf_idf, key=lambda x: x[1], reverse=True)
true_meaning = sort_array[0][0].split('_')[1]
print('\n经过词义消岐，%s在该句子中的意思为 %s .' % (wsd_word, true_meaning))

# # Train a Naive Bayes model using the CountVectorizer features
# nb_count = MultinomialNB()
# nb_count.fit(X_train_count, y_train)

# # Train a Naive Bayes model using the TfidfVectorizer features
# nb_tfidf = MultinomialNB()
# nb_tfidf.fit(X_train_tfidf, y_train)

# # Train a Decision Tree model using the CountVectorizer features
# dt_count = DecisionTreeClassifier()
# dt_count.fit(X_train_count, y_train)

# # Train a Decision Tree model using the TfidfVectorizer features
# dt_tfidf = DecisionTreeClassifier()
# dt_tfidf.fit(X_train_tfidf, y_train)

# # Train a SVM model using the CountVectorizer features
# svm_count = SVC()
# svm_count.fit(X_train_count, y_train)

# # Train a SVM model using the TfidfVectorizer features
# svm_tfidf = SVC()
# svm_tfidf.fit(X_train_tfidf, y_train)

# # Train a Random Forest model using the CountVectorizer features
# rf_count = RandomForestClassifier()
# rf_count.fit(X_train_count, y_train)

# # Train a Random Forest model using the TfidfVectorizer features
# rf_tfidf = RandomForestClassifier()
# rf_tfidf.fit(X_train_tfidf, y_train)

# # Train a MLP model using the CountVectorizer features
# mlp_count = MLPClassifier()
# mlp_count.fit(X_train_count, y_train)

# # Train a MLP model using the TfidfVectorizer features
# mlp_tfidf = MLPClassifier()
# mlp_tfidf.fit(X_train_tfidf, y_train)
