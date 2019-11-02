import tensorflow as tf
from tensorflow.keras import layers
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print(tf.__version__)

mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

# hyperparameters
TRAIN_SPLIT_RATIO = 0.9
BATCH_SIZE = 512
BUFFER_SIZE = 10000
PAST_HISTORY = 120
FUTURE_TARGET = 24
STEP = 3
EPOCHS = 5
LEARNING_RATE = 0.0005 # default: 0.001, optimal: 0.0005 (apply learning rate decay when near convergence)
CLIP_VALUE = 1.0 # apply gradient clipping to prevent exploding gradients

# dataset path
FILE_NAME = 'data/sfo_6_features.csv'
# weights path 
checkpoint_path = 'weights_big_nn/W'

# control flow: train, evaluate, predict
train = False
evaluate = False
predict = False

# dataset functions
def univariate_data(dataset, start_index, end_index, history_size, target_size):
    data, labels = list(), list()

    start_index += history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i)
        # reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i + target_size])
    return np.array(data), np.array(labels)

def multivariate_data(dataset, target, start_index, end_index, history_size, target_size, step, single_step = False):
    data, labels = list(), list()

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step) # step represents the interval at which the past history data is sampled for training
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i + target_size])
        else:
            labels.append(target[i: i + target_size])

    return np.array(data), np.array(labels)

# visualize example
def create_time_steps(length):
    time_steps = list()
    for i in range(-length, 0, 1):
        time_steps.append(i)
    return time_steps

def show_plot(plot_data, delta, title):
    labels = ['History', 'True Future', 'Model Prediction']
    marker = ['.-', 'rx', 'go']
    time_steps = create_time_steps(plot_data[0].shape[0])
    if delta:
        future = delta
    else:
        future = 0

    plt.title(title)
    for i, x in enumerate(plot_data):
        if i:
            plt.plot(future, plot_data[i], marker[i], markersize = 10, label = labels[i])
        else:
            plt.plot(time_steps, plot_data[i].flatten(), marker[i], label = labels[i])
    plt.legend()
    plt.xlim([time_steps[0], (future + 10) * 4])
    plt.xlabel('Time Step')
    return plt

def plot_train_history(history, title = "Learning Curve"):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    plt.figure()
    plt.plot(epochs, loss, 'b', label = 'Training loss')
    plt.plot(epochs, val_loss, 'r', label = 'Validation loss')
    plt.title(title)
    plt.legend()
    plt.show()

def multi_step_plot(history, true_future, prediction, mae_metric):
    plt.figure(figsize = (12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 1]), 'g-', label = 'History')
    plt.plot(np.arange(num_out) / STEP, np.array(true_future), 'b--', label = 'True Future')
    if prediction.any():
        plt.plot(np.arange(num_out) / STEP, np.array(prediction), 'r--', label = 'Predicted Future')
    plt.title('Past history, True future, Predicted future, MAE metric: ' + str(mae_metric))
    plt.legend()
    plt.show()

# create baseline: average of past history
def baseline(history):
    return np.mean(history)

# undo data normalization
def unnormalize(output_data, data_mean, data_std, feature_index):
    return output_data * data_std[feature_index] + data_mean[feature_index]

# load file
# fname = 'data/' + AIRPORT + '_6_features.csv' # valid is CA local time
fname = FILE_NAME
df = pd.read_csv(fname)
print(df.head())
print('dataframe size: {}'.format(df.size))
print('dataframe shape" {}'.format(df.shape))
print('dataframe ndim: {}'.format(df.ndim))

# features: tmpc (temperature in C), dwpc (dew point in C), relh (relative humidity in %),
# sped (wind speed in mph), alti (altimeter in inches (air pressure)), vsby (visibility in miles)
features_considered = ['tmpc', 'dwpc', 'relh', 'sped', 'alti', 'vsby']
features = df[features_considered]
features.index = df['valid']
print(features.head())

# visualize
# features.plot(subplots = True)
# plt.show()

# normalize dataset
dataset = features.values
data_mean = dataset.mean(axis = 0)
data_std = dataset.std(axis = 0)
dataset = (dataset - data_mean) / data_std
print(dataset[:10])
print(dataset.shape)
tmpc_data = dataset[:, 0] # get tmpc_sfo column
print(tmpc_data[:10])

# calculate train_split
TRAIN_SPLIT = int(dataset.shape[0] * TRAIN_SPLIT_RATIO)

# create train and val datasets: this step is time consuming and expensive on RAM
x_train_multi, y_train_multi = multivariate_data(
    dataset, tmpc_data, 0, TRAIN_SPLIT,
    PAST_HISTORY, FUTURE_TARGET, STEP
)
x_val_multi, y_val_multi = multivariate_data(
    dataset, tmpc_data, TRAIN_SPLIT, None,
    PAST_HISTORY, FUTURE_TARGET, STEP
)
print('\nSingle window of past history: {}'.format(x_train_multi[0].shape))
print('Target temperature to predict: {}'.format(y_train_multi[0].shape))
print('Incoming train data shape: {}'.format(x_train_multi.shape))
print('Incoming 1 sample of train data shape: {}\n'.format(x_train_multi.shape[-2:]))

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat() # I shuffled validation data too!

# plot a sample
# for x, y in train_data_multi.take(1):
#     multi_step_plot(x[0], y[0], np.array([0]))

# create model
model = tf.keras.models.Sequential([
    # layers.Lambda(lambda x: tf.expand_dims(x, axis = -1), input_shape = x_train_multi.shape[-2:]),
    # layers.Conv2D(filters = 60, kernel_size = 3, strides = 1, activation = 'relu'),
    layers.LSTM(128, activation = 'tanh', return_sequences = True, input_shape = x_train_multi.shape[-2:]),
    layers.Dropout(0.2),
    layers.LSTM(128, activation = 'tanh', return_sequences = True),
    layers.Dropout(0.2),
    layers.LSTM(128, activation = 'tanh', return_sequences = True),
    layers.Dropout(0.2),
    layers.LSTM(128, activation = 'tanh', return_sequences = True),
    layers.Dropout(0.2),
    layers.LSTM(128, activation = 'relu'),
    layers.Dropout(0.2),
    layers.Dense(1024, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(512, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(256, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(128, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(64, activation = 'relu', kernel_regularizer = tf.keras.regularizers.l2(0.001)),
    layers.Dropout(0.2),
    layers.Dense(FUTURE_TARGET)
], name = 'Time_Series_Forecast_LSTM_Model')
model.summary()

# compile model
model.compile(
    # optimizer = tf.keras.optimizers.RMSprop(clipvalue = 1.0),
    # optimizer = 'adam',
    # optimizer = tf.keras.optimizers.SGD(lr = 1e-8, momentum = 0.9),
    optimizer = tf.keras.optimizers.Adam(learning_rate = LEARNING_RATE, clipvalue = CLIP_VALUE), # default = 0.001, optimal = 0.008
    # loss = 'mae'
    loss = tf.keras.losses.Huber()
)

# for x, y in val_data_multi.take(1):
#     print(model.predict(x).shape)

# define checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only = True)
try:
    model.load_weights(checkpoint_path)
    print('Weights detected.')
except:
    print('No weights detected.')

# define learning rate schedule callback
lr_schedule = tf.keras.callbacks.LearningRateScheduler(
    lambda epoch: 0.00001 * 10**(epoch / 20))

# train model
if train:
    history = model.fit(
        train_data_multi,
        epochs = EPOCHS,
        steps_per_epoch = x_train_multi.shape[0] // BATCH_SIZE,
        # steps_per_epoch = 200,
        validation_data = val_data_multi,
        validation_steps = x_val_multi.shape[0] // BATCH_SIZE,
        # validation_steps = 50,
        callbacks = [cp_callback]
    )
    plot_train_history(history, 'Multi-Step Train and Validation Loss')

# def optimize_learning_rate(history):
#     # find optimal learning rate
#     plt.semilogx(history.history["lr"], history.history["loss"])
#     plt.axis([0.00001, 1, 0.2, 0.5])
#     plt.xlabel('Learning Rate')
#     plt.ylabel('Loss')
#     plt.title('Learning Rate vs Loss')
#     plt.show()
# optimize_learning_rate(history) # best learning rate for Adam is 0.008 (default = 0.001)

# evaluate model
if evaluate:
    model.evaluate(val_data_multi, steps = x_val_multi.shape[0] // BATCH_SIZE)

# predict the future
if predict:
    for x, y in val_data_multi.take(10):
        # undo data normalization
        unnormalized_x = unnormalize(x[0], data_mean, data_std, 0)
        unnormalized_y = unnormalize(y[0], data_mean, data_std, 0)
        # predict and unnormalize
        unnormalized_pred = unnormalize(model.predict(x)[0], data_mean, data_std, 0)
        # calculate mae metric of real output values
        mae_metric = tf.keras.metrics.mean_absolute_error(unnormalized_y, unnormalized_pred).numpy()
        # plot predictions
        multi_step_plot(
            unnormalized_x,
            unnormalized_y,
            unnormalized_pred,
            mae_metric
        )
