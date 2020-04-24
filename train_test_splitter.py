import os,re
import numpy as np
import pandas as pd

# split into train and test
# the ratio is about 3:1
# set random state with seed 71
random_state = np.random.RandomState(71)
def train_test_split(x):
    global random_state
    m = len(x)
    return random_state.choice(x.index, m // 5, replace=False)

def train_test_splitter():
    # load csv files from data directory
    infiles = os.listdir("data\\")
    goodbook = {}
    for tab in infiles:
        tab_name = re.search(r"([^\.]*)\.csv", tab)
        if tab_name is not None:
            goodbook[tab_name.group(1)] = pd.read_csv("data\\" + tab_name.group(0))

    ratings = goodbook["ratings"]
    print("There are {} records in total.".format(ratings.shape[0]))

    # check all the duplicates in ratings
    ratings_dupl = ratings.loc[ratings.drop("rating",1).duplicated(keep=False)]
    print("There are {} duplicates in the data.".format(ratings_dupl.shape[0]))

    # see if any user rates more than once and if their multiple ratings are the same
    is_allsame = ratings_dupl.groupby(["book_id","user_id"]).agg(lambda x:x.nunique()==1)
    print("Are records from the same user-book pair are also the same: {}.".format(np.all(is_allsame)))
    # we will abort all the records related to duplication
    ratings_fixed = ratings.drop(labels=ratings_dupl.index)
    print("duplicates removed")

    # check the minimal times of rating as a user
    rating_per_user = ratings_fixed.groupby("user_id").agg({"book_id":"count"})
    print("Minimal times of rating among all the users is/are: {}.".format(rating_per_user["book_id"].min()))

    # check the minimal times being rated of a book
    rated_per_book = ratings_fixed.groupby("book_id").agg({"user_id":"count"})
    print("Minimal times of being rated among all the books is/are: {}.".format(rated_per_book["user_id"].min()))

    # filter out users that only rates less than or equal to 5
    users_valid = ratings_fixed.groupby("user_id").agg({"book_id":"count"}).query("book_id>5")
    ratings_valid = ratings_fixed.set_index("user_id").loc[users_valid.index].reset_index()
    print("There remain {} valid records.".format(ratings_valid.shape[0]))

    test_indices = ratings_valid.groupby("user_id").apply(train_test_split)
    test_indices = pd.Index(np.hstack(test_indices))
    ratings_test = ratings_valid.loc[test_indices]
    ratings_train = ratings_valid.loc[ratings_valid.index.difference(test_indices)]

    # check if processed directory exists under current path
    if not os.path.exists("processed"):
        os.mkdir("processed")
    # save as npz compressed
    ratings_matrix = ratings_train.pivot(index="user_id",columns="book_id",values="rating").to_numpy()
    np.savez_compressed("processed\\ratings_train",ratings_train=ratings_matrix)
    ratings_matrix = ratings_test.pivot(index="user_id",columns="book_id",values="rating").to_numpy()
    np.savez_compressed("processed\\ratings_test",ratings_test=ratings_matrix)

if __name__ == "__main__":
    train_test_splitter()