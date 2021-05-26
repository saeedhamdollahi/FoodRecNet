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

# get ingredients
cur.execute("SELECT * FROM public.ingredient ORDER BY index ASC")
ingredients = cur.fetchall()
ingredients_count = len(ingredients)
ingredients_dict = {}
for ingredient in ingredients:
    ingredients_dict[ingredient['ingredient_id']] = {'index': feature_index_counter}
    feature_dict[feature_index_counter] = {'type': 2, 'id': ingredient['ingredient_id']}
    feature_index_counter += 1

# get max amounts
cur.execute("SELECT max(time) FROM public.recipe")
max_time = cur.fetchone()[0]
cur.execute("SELECT max(servings) FROM public.recipe")
max_servings = cur.fetchone()[0]
cur.execute("SELECT max(steps) FROM public.recipe")
max_steps = cur.fetchone()[0]
cur.execute("SELECT max(ingredient_count) FROM public.recipe")
max_ingredient_count = cur.fetchone()[0]
cur.execute("SELECT max(calorie) FROM public.recipe")
max_calorie = cur.fetchone()[0]

# get recipes
cur.execute("SELECT * FROM public.recipe ORDER BY index ASC")
recipes = cur.fetchall()
recipes_count = len(recipes)
main_counter = 1
for recipe in recipes:
    # recipe vector
    recipe_vector = []
    for i in range(feature_index_counter):
        recipe_vector.append(0)

    # get recipe categories
    cur.execute("SELECT * FROM public.recipe_category_map WHERE recipe_id = %s", (recipe['recipe_id'],))
    recipe_categories = cur.fetchall()
    for recipe_category in recipe_categories:
        recipe_category_index = categories_dict[recipe_category['category_id']]['index']
        recipe_vector[recipe_category_index] = 1

    # get recipe ingredients
    cur.execute("SELECT * FROM public.recipe_ingredient_map WHERE recipe_id = %s", (recipe['recipe_id'],))
    recipe_ingredients = cur.fetchall()
    for recipe_ingredient in recipe_ingredients:
        recipe_ingredient_index = ingredients_dict[recipe_ingredient['ingredient_id']]['index']
        recipe_vector[recipe_ingredient_index] = 1

    # save vector
    recipe_vector_json = ','.join([str(n) for n in recipe_vector])

    # recipe second vector
    recipe_second_vector = []
    recipe_second_vector.append(round(recipe['time'] / max_time, 3))
    recipe_second_vector.append(round(recipe['servings'] / max_servings, 3))
    recipe_second_vector.append(round(recipe['steps'] / max_steps, 3))
    recipe_second_vector.append(round(recipe['ingredient_count'] / max_ingredient_count, 3))
    recipe_second_vector.append(round(recipe['calorie'] / max_calorie, 3))

    # save vector
    recipe_second_vector_str = ','.join([str(n) for n in recipe_second_vector])

    # update recipe vector
    cur.execute("UPDATE public.recipe SET vector = %s, second_vector = %s WHERE recipe_id = %s", (recipe_vector_json, recipe_second_vector_str, recipe['recipe_id'],))

    if main_counter % 100 == 0:
        conn.commit()

    main_counter += 1
    print(main_counter, recipe['recipe_id'])

conn.commit()
