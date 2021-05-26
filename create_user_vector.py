import numpy as np
import pandas as pd
import json
import math
import psycopg2
import psycopg2.extras


# connect to database
conn = psycopg2.connect(host="", database="", user="", password="")
cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

# features
feature_index_counter = 0
feature_dict = {}

# get categories
cur.execute("SELECT * FROM public.category ORDER BY index ASC")
categories = cur.fetchall()
categories_count = len(categories)
categories_dict = {}
for category in categories:
    categories_dict[category['category_id']] = {'index': feature_index_counter}
    feature_dict[feature_index_counter] = {'type': 1, 'id': category['category_id']}
    feature_index_counter += 1
del categories

# get ingredients
cur.execute("SELECT * FROM public.ingredient ORDER BY index ASC")
ingredients = cur.fetchall()
ingredients_count = len(ingredients)
ingredients_dict = {}
for ingredient in ingredients:
    ingredients_dict[ingredient['ingredient_id']] = {'index': feature_index_counter}
    feature_dict[feature_index_counter] = {'type': 2, 'id': ingredient['ingredient_id']}
    feature_index_counter += 1
del ingredients

# get recipes
cur.execute("SELECT recipe_id, vector FROM public.recipe ORDER BY recipe_id")
recipes = cur.fetchall()
recipes_count = len(recipes)
recipes_dict = {}
for recipe_row in recipes:
    recipes_dict[recipe_row['recipe_id']] = recipe_row['vector']
del recipes

# get users
cur.execute("SELECT user_id FROM public.user ORDER BY user_id ASC")
users = cur.fetchall()
main_counter = 1
for user in users:
    # user vector
    user_vector = []
    for i in range(feature_index_counter):
        user_vector.append(0)

    # get user reviews
    cur.execute("SELECT recipe_id, rate FROM public.review WHERE user_id = %s", (user['user_id'],))
    reviews = cur.fetchall()
    user_reviews_count = len(reviews)

    # create vector
    for review in reviews:
        # get review recipe vector
        review_recipe_vector = [int(n) for n in recipes_dict[review['recipe_id']].split(',')]

        # bipartite rate
        bipartite_rate = 0
        if review['rate'] == 5:
            bipartite_rate = 1
        elif review['rate'] == 4:
            bipartite_rate = 0.5
        elif review['rate'] == 3:
            bipartite_rate = 0
        elif review['rate'] == 2:
            bipartite_rate = -0.5
        else:
            bipartite_rate = -1

        # update user vector
        for i in range(feature_index_counter):
            recipe_vector_value = review_recipe_vector[i]
            if recipe_vector_value:
                user_vector[i] += (bipartite_rate * recipe_vector_value)

        # max value
        max_value = 0
        for i in range(feature_index_counter):
            abs_value = math.fabs(user_vector[i])
            if abs_value > max_value:
                max_value = abs_value

        # normalize
        if max_value > 0:
            for i in range(feature_index_counter):
                user_vector[i] = round(user_vector[i] / max_value, 3)

    # save vector
    user_vector_str = ','.join([str(n) for n in user_vector])

    # update user vector
    cur.execute("UPDATE public.user SET vector = %s WHERE user_id = %s", (user_vector_str, user['user_id'],))

    if main_counter % 100 == 0:
        conn.commit()

    main_counter += 1
    print(main_counter, user['user_id'])

conn.commit()
