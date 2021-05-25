import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate
from keras.preprocessing import image
import keras.backend as K


# train data generator
def train_data_generator(batch_size=100):
    reader = pd.read_csv("train.txt", delimiter="#", low_memory=False, chunksize=batch_size, dtype=np.str)
    while True:
        try:
            chunk = next(reader)
        except StopIteration:
            reader = pd.read_csv("train.txt", delimiter="#", low_memory=False, chunksize=batch_size, dtype=np.str)
            chunk = next(reader)
        chunk = chunk.to_numpy(dtype=np.str)
        chunk_x_1 = []
        chunk_x_2 = []
        chunk_x_3 = []
        chunk_x_4 = []
        chunk_x_5 = []
        chunk_y = []
        for row in chunk:
            recipe_image = image.load_img("images/{}.jpg".format(row[7]))
            recipe_image = image.img_to_array(recipe_image)
            recipe_image = recipe_image.astype('float32') / 255

            chunk_x_1.append([float(item) for item in row[0].split(',')])
            chunk_x_2.append([float(item) for item in row[1].split(',')])
            chunk_x_3.append([float(item) for item in row[3].split(',')])
            chunk_x_4.append(recipe_image)
            chunk_x_5_temp_1 = row[2].split(',')
            chunk_x_5_temp_2 = row[4].split(',')
            chunk_x_5.append([float(chunk_x_5_temp_1[0]), float(chunk_x_5_temp_1[1]), float(chunk_x_5_temp_1[2]),
                              float(chunk_x_5_temp_1[3]), float(chunk_x_5_temp_1[4]), float(chunk_x_5_temp_2[0]),
                              float(chunk_x_5_temp_2[1])])
            chunk_y.append(float(row[5]))
        yield [np.array(chunk_x_1), np.array(chunk_x_2), np.array(chunk_x_3), np.array(chunk_x_4), np.array(chunk_x_5)], np.array(chunk_y)


# test data generator
def test_data_generator(batch_size=100):
    reader = pd.read_csv("test.txt", delimiter="#", low_memory=False, chunksize=batch_size, dtype=np.str)
    while True:
        try:
            chunk = next(reader)
        except StopIteration:
            reader = pd.read_csv("test.txt", delimiter="#", low_memory=False, chunksize=batch_size, dtype=np.str)
            chunk = next(reader)
        chunk = chunk.to_numpy(dtype=np.str)
        chunk_x_1 = []
        chunk_x_2 = []
        chunk_x_3 = []
        chunk_x_4 = []
        chunk_x_5 = []
        chunk_y = []
        for row in chunk:
            recipe_image = image.load_img("images/{}.jpg".format(row[7]))
            recipe_image = image.img_to_array(recipe_image)
            recipe_image = recipe_image.astype('float32') / 255

            chunk_x_1.append([float(item) for item in row[0].split(',')])
            chunk_x_2.append([float(item) for item in row[1].split(',')])
            chunk_x_3.append([float(item) for item in row[3].split(',')])
            chunk_x_4.append(recipe_image)
            chunk_x_5_temp_1 = row[2].split(',')
            chunk_x_5_temp_2 = row[4].split(',')
            chunk_x_5.append([float(chunk_x_5_temp_1[0]), float(chunk_x_5_temp_1[1]), float(chunk_x_5_temp_1[2]),
                              float(chunk_x_5_temp_1[3]), float(chunk_x_5_temp_1[4]), float(chunk_x_5_temp_2[0]),
                              float(chunk_x_5_temp_2[1])])
            chunk_y.append(float(row[5]))
        yield [np.array(chunk_x_1), np.array(chunk_x_2), np.array(chunk_x_3), np.array(chunk_x_4), np.array(chunk_x_5)], np.array(chunk_y)


# rmse
def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))


# get data
batch_size = 32
user_input_dim = 4567
recipe_features_input_dim = 4567
recipe_text_input_dim = 5000
recipe_image_input_dim = (240, 320, 3)
other_features_input_dim = 7
num_total_samples = 2975564

num_training_samples = int(num_total_samples * 0.8)
num_testing_samples = num_total_samples - num_training_samples

train_data_generator_iterator = train_data_generator(batch_size)
test_data_generator_iterator = test_data_generator(batch_size)
test_data_generator_iterator2 = test_data_generator(batch_size)

# create model
# user
user_model_input = Input(shape=(user_input_dim,))
user_model = Dense(units=1024, activation='relu')(user_model_input)
user_model = Dense(units=512, activation='relu')(user_model)
user_model = Dense(units=256, activation='relu')(user_model)
user_model = Dense(units=128, activation='relu')(user_model)

# recipe features
recipe_features_model_input = Input(shape=(recipe_features_input_dim,))
recipe_features_model = Dense(units=1024, activation='relu')(recipe_features_model_input)
recipe_features_model = Dense(units=512, activation='relu')(recipe_features_model)
recipe_features_model = Dense(units=256, activation='relu')(recipe_features_model)
recipe_features_model = Dense(units=128, activation='relu')(recipe_features_model)

# recipe text
recipe_text_model_input = Input(shape=(recipe_text_input_dim,))
recipe_text_model = Dense(units=1024, activation='relu')(recipe_text_model_input)
recipe_text_model = Dense(units=512, activation='relu')(recipe_text_model)
recipe_text_model = Dense(units=256, activation='relu')(recipe_text_model)
recipe_text_model = Dense(units=128, activation='relu')(recipe_text_model)

# recipe image
recipe_image_model_input = Input(shape=(recipe_image_input_dim[0], recipe_image_input_dim[1], recipe_image_input_dim[2],))
recipe_image_model = Conv2D(32, (3, 3), activation='relu')(recipe_image_model_input)
recipe_image_model = MaxPooling2D((2, 2))(recipe_image_model)
recipe_image_model = Conv2D(64, (3, 3), activation='relu')(recipe_image_model)
recipe_image_model = MaxPooling2D((2, 2))(recipe_image_model)
recipe_image_model = Conv2D(128, (3, 3), activation='relu')(recipe_image_model)
recipe_image_model = MaxPooling2D((2, 2))(recipe_image_model)
recipe_image_model = Conv2D(256, (3, 3), activation='relu')(recipe_image_model)
recipe_image_model = MaxPooling2D((2, 2))(recipe_image_model)
recipe_image_model = Conv2D(256, (3, 3), activation='relu')(recipe_image_model)
recipe_image_model = MaxPooling2D((2, 2))(recipe_image_model)
recipe_image_model = Flatten()(recipe_image_model)
recipe_image_model = Dense(units=1024, activation='relu')(recipe_image_model)
recipe_image_model = Dense(units=512, activation='relu')(recipe_image_model)
recipe_image_model = Dense(units=256, activation='relu')(recipe_image_model)
recipe_image_model = Dense(units=128, activation='relu')(recipe_image_model)

# recipe model
recipe_model = concatenate([recipe_features_model, recipe_text_model, recipe_image_model])
recipe_model = Dense(units=256, activation='relu')(recipe_model)
recipe_model = Dense(units=128, activation='relu')(recipe_model)

# other
other_features_input = Input(shape=(other_features_input_dim,))

# main model
main_model = concatenate([user_model, recipe_model, other_features_input])
main_model = Dense(units=256, activation='relu')(main_model)
main_model = Dense(units=128, activation='relu')(main_model)
main_model = Dense(units=64, activation='relu')(main_model)
main_model = Dense(units=1, activation='linear')(main_model)
model = Model(inputs=[user_model_input, recipe_features_model_input, recipe_text_model_input, recipe_image_model_input, other_features_input], outputs=main_model)
model.summary()

# compile
model.compile(
    optimizer='adam',
    loss=rmse,
    metrics=['mse', 'mae', 'mape', rmse]
)

# fit
model.fit_generator(
    generator=train_data_generator_iterator,
    steps_per_epoch=(num_training_samples // batch_size) + 1,
    epochs=3,
    verbose=1,
)

# save
model.save_weights('architecture_5.h5')

# evaluate
metrics = model.evaluate_generator(
    generator=test_data_generator_iterator,
    steps=(num_testing_samples // batch_size)+1,
    verbose=1,
)
print('metrics:', metrics)

# predict
prediction = model.predict_generator(
    generator=test_data_generator_iterator2,
    steps=(num_testing_samples // batch_size) + 1,
    verbose=1,
)
print('prediction:', prediction)
pd.DataFrame(prediction).to_csv("architecture_5.csv", index=False)
