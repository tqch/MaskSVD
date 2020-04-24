from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

test_ratio = 0.2
data_path = 'team_data\\'

ratings = pd.read_csv(data_path + 'ext_ratings.csv')
user = pd.read_csv(data_path + 'reg_user.csv', index_col=0)
user['user_group'] = user['user_id'] - np.round(user['user_id'] / 10) * 10

sub_user = user[(user['range_rating'] > 0) & (user['count_rating'] > 5)]
# sub_user = user[(user['max_rating'] == 5) & (user['count_rating'] > 5)]
sub_user = sub_user[(sub_user['user_group'] == 1)]

sub_user = sub_user[['user_id', 'count_rating', 'average_rating', 'max_rating', 'min_rating', 'range_rating']]
sub_user.rename(columns={'average_rating': 'average_rating_user'}, inplace=True)

book = pd.read_csv(data_path + 'reg_books.csv')

book = book[['id', 'books_count', 'author_num', 'language_code', 'average_rating', 'ratings_count', 'work_text_reviews_count', 'ratings_1', 'ratings_2', 'ratings_3', 'ratings_4', 'ratings_5']]
book.rename(columns={'average_rating': 'average_rating_book'}, inplace=True)
book.rename(columns={'id': 'book_id'}, inplace=True)

ext_rating = ratings.merge(sub_user, on='user_id', how='inner')
ext_rating = ext_rating.merge(book, on='book_id', how='inner')

def mlr(X_train,y_train,X_test,y_test):
    n_features = X_train.shape[1]

    x = tf.placeholder(tf.float32, shape=[None, n_features])
    y = tf.placeholder(tf.float32, shape=[None])

    m = 10
    learning_rate = 0.1
    u = tf.Variable(tf.random_normal([n_features, m], 0.0, 0.1), name='u')
    w = tf.Variable(tf.random_normal([n_features, m], 0.0, 0.1), name='w')

    U = tf.matmul(x, u)
    p1 = tf.nn.softmax(U)

    W = tf.matmul(x, w)
    p2 = tf.nn.sigmoid(W)

    pred = tf.reduce_sum(tf.multiply(p1, p2), 1)

    cost = tf.reduce_mean(-y * tf.log(pred) - (1 - y) * tf.log(1 - pred))
    train_op = tf.train.FtrlOptimizer(learning_rate).minimize(cost)

    time_s = time.time()
    result = []
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(0, 10):
            f_dict = {x: X_train, y: y_train}

            _, cost_, predict_ = sess.run([train_op, cost, pred], feed_dict=f_dict)

            acc = np.mean(y_train == (predict_ >= 0.5))

            time_t = time.time()

            f_dict = {x: X_test, y: y_test}
            predict_test = sess.run(pred, feed_dict=f_dict)

            test_acc = np.mean(y_test == (predict_test >= 0.5))
            print("%d epoch, %ld s, train loss:%f, train_acc:%f, test_acc:%f" % (
            epoch, (time_t - time_s), cost_, acc, test_acc))

X, y = ext_rating[['user_id', 'book_id']], ext_rating['is_like']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1)

encoder = OneHotEncoder(handle_unknown='ignore').fit(X_train)

X_train = encoder.transform(X_train).todense()
X_test = encoder.transform(X_test).todense()

print("纯ID")
mlr(X_train,y_train,X_test,y_test)
print()

X, y = ext_rating[['user_id', 'book_id','count_rating','average_rating_user', 'range_rating', 'average_rating_book']], ext_rating['is_like']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1)
# PCA on dense features
pca = PCA(whiten=True)
X_orth_train = X_train.drop(['user_id', 'book_id'], axis=1)
X_orth_test = X_test.drop(['user_id', 'book_id'], axis=1)
pca.fit(X_orth_train)
X_orth_train = pca.transform(X_orth_train)
X_orth_train = pd.DataFrame(X_orth_train)
ids = X_train[['user_id', 'book_id']]
ids.reset_index(inplace=True, drop=True)
X_train = pd.concat([ids, X_orth_train], axis=1)
X_orth_test = pca.transform(X_orth_test)
X_orth_test = pd.DataFrame(X_orth_test)
ids = X_test[['user_id', 'book_id']]
ids.reset_index(inplace=True, drop=True)
X_test = pd.concat([ids, X_orth_test], axis=1)
# One-hot on id features
ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), [0, 1])], remainder='passthrough').fit(X_train)
X_train = ct.transform(X_train).todense()
X_test = ct.transform(X_test).todense()

print("ID + 一些手动挑的特征")
mlr(X_train,y_train,X_test,y_test)
print()

X, y = ext_rating.drop(['is_like', 'rating', 'language_code'], axis=1), ext_rating['is_like']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_ratio, random_state=1)
# PCA on dense features
pca = PCA(whiten=True)
X_orth_train = X_train.drop(['user_id', 'book_id'], axis=1)
X_orth_test = X_test.drop(['user_id', 'book_id'], axis=1)
pca.fit(X_orth_train)
X_orth_train = pca.transform(X_orth_train)
X_orth_train = pd.DataFrame(X_orth_train)
ids = X_train[['user_id', 'book_id']]
ids.reset_index(inplace=True, drop=True)
X_train = pd.concat([ids, X_orth_train], axis=1)
X_orth_test = pca.transform(X_orth_test)
X_orth_test = pd.DataFrame(X_orth_test)
ids = X_test[['user_id', 'book_id']]
ids.reset_index(inplace=True, drop=True)
X_test = pd.concat([ids, X_orth_test], axis=1)
# One-hot on id features
ct = ColumnTransformer( [('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'), [0, 1])], remainder='passthrough').fit(X_train)
X_train = ct.transform(X_train).todense()
X_test = ct.transform(X_test).todense()

print("ID + 尽量多的特征")
mlr(X_train,y_train,X_test,y_test)
print()
