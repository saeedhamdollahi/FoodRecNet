import re
import json
import math
import psycopg2
import psycopg2.extras
import os.path
from keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords

# connect to database
conn = psycopg2.connect(host="", database="", user="", password="")
#cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
cur = conn.cursor()


def remove_stop_words(train_sentences, stop_words):
    for i, sentence in enumerate(train_sentences):
        sentence = re.sub(r'[0-9]+', '', sentence.lower())
        new_sent = [word for word in sentence.split() if word not in stop_words]
        train_sentences[i] = ' '.join(new_sent)
    return train_sentences

recipe_ids = []
train_sentences = []
cur.execute("SELECT recipe_id, instructions FROM public.recipe WHERE removed = false ORDER BY recipe_id ASC")
recipes = cur.fetchall()
for recipe_row in recipes:
    recipe_ids.append(recipe_row[0])
    train_sentences.append(recipe_row[1])

train_sentences = remove_stop_words(train_sentences, set(stopwords.words("english")))
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_sentences_matrix = tokenizer.texts_to_matrix(train_sentences, mode='binary')

word_count = len(train_sentences_matrix[0])
recipe_count = len(train_sentences_matrix)

for i in range(recipe_count):
    recipe_id = recipe_ids[i]
    recipe_text = train_sentences_matrix[i]
    recipe_text_vector = []
    for j in range(word_count):
        recipe_text_vector.append(0)
    for j in range(word_count+1):
        if j != 0:
            recipe_text_vector[j-1] = int(train_sentences_matrix[i][j-1])

    # save vector
    recipe_text_vector_str = ','.join([str(n) for n in recipe_text_vector])

    # update recipe vector
    cur.execute("UPDATE public.recipe SET text_vector = %s WHERE recipe_id = %s", (recipe_text_vector_str, recipe_id,))

    if i % 100 == 0:
        conn.commit()

    print(i, recipe_id)

conn.commit()
