import re
import json
import math
import psycopg2
import psycopg2.extras
import os.path


# connect to database
conn = psycopg2.connect(host="", database="", user="", password="")
#cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
cur = conn.cursor()

# get recipes
cur.execute("SELECT recipe_id, vector, second_vector, text_vector FROM public.recipe ORDER BY recipe_id")
recipes = cur.fetchall()
recipes_count = len(recipes)
recipes_dict = {}
for recipe_row in recipes:
    recipes_dict[recipe_row[0]] = {'vector': recipe_row[1], 'second_vector': recipe_row[2], 'text_vector': recipe_row[3]}
del recipes
print(recipes_count)

# get users
cur.execute("SELECT user_id, vector FROM public.user ORDER BY user_id")
users = cur.fetchall()
users_count = len(users)
users_dict = {}
for user_row in users:
    users_dict[user_row[0]] = user_row[1]
del users
print(users_count)

# initialize
f = open("all.txt", "a")
batch_size = 1000
low_index = 1
high_index = batch_size

# iterate
counter = 0
while True:
    # get data
    cur.execute("SELECT user_id, recipe_id, rate FROM public.review WHERE review_id BETWEEN %s AND %s ORDER BY review_id ASC", (low_index, high_index,))
    reviews = cur.fetchall()

    # update index
    low_index += batch_size
    high_index += batch_size
    if len(reviews) == 0:
        break

    # write file
    for review_row in reviews:
        # get users
        cur.execute("SELECT vector, average_rate FROM public.user WHERE user_id = %s", (review_row[0],))
        user_row = cur.fetchone()

        # get recipes
        cur.execute("SELECT vector, second_vector, text_vector, average_rate FROM public.recipe WHERE recipe_id = %s", (review_row[1],))
        recipe_row = cur.fetchone()

        f.write(
            "{}#{}#{}#{}#{}#{}#{}#{}\n".format(
                user_row[0],
                recipe_row[0],
                recipe_row[1],
                recipe_row[2],
                "{},{}".format(str(round(user_row[1] / 5, 3)), str(round(recipe_row[3] / 5, 3)),),
                str(review_row[2]),
                str(review_row[0]),
                str(review_row[1])
            )
        )

        counter += 1
        print(counter)

    del reviews

f.close()

num_total_samples = 2975564
num_training_samples = int(num_total_samples * 0.8)
num_testing_samples = num_total_samples - num_training_samples
f = open("all.txt", "r")
f_train = open("train.txt", "a")
f_test = open("test.txt", "a")
counter = 0
for line in f:
    counter += 1
    if counter <= num_training_samples:
        f_train.write(line)
    else:
        f_test.write(line)
print(counter)
