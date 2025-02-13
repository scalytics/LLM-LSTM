#LSTM Model for time series forecast, (c) infinimesh, Scalytics (www.scalytics.io) and affiliates, 2020 - 2023
# Apache License 2.0
#Some functions were copied from TensforFlow website time-series tutorial, see: https://www.tensorflow.org/tutorials/structured_data/time_series#top_of_page
#GitHub: https://github.com/tensorflow/docs/blob/master/site/en/tutorials/structured_data/time_series.ipynb
#-----------------------------------

import os
import datetime
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import IPython
import IPython.display
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import datetime as dt
from sklearn.preprocessing import MinMaxScaler
mpl.rcParams['figure.figsize'] = (8, 6)
mpl.rcParams['axes.grid'] = False

import warnings
warnings.filterwarnings("ignore")
from tensorflow.python.client import device_lib 

#Some settings
strategy = tf.distribute.MirroredStrategy()
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(device_lib.list_local_devices())
tf.keras.backend.set_floatx('float64')

for chunk in pd.read_csv("smartmeter.csv", chunksize= 10**6):
    print(chunk)

data = pd.DataFrame(chunk)
data = data.drop(['device_id', 'device_name', 'property'], axis = 1)

# Creating daytime input
def time_d(x):
    k = datetime.datetime.strptime(x, "%H:%M:%S")
    y = k - datetime.datetime(1900, 1, 1)
    return y.total_seconds()

daytime = data['timestamp'].str.slice(start = 11 ,stop=19)
secondsperday = daytime.map(lambda i: time_d(i))
data['timestamp'] = data['timestamp'].str.slice(stop=19)
data['timestamp'] = data['timestamp'].map(lambda i: dt.datetime.strptime(i, '%Y-%m-%d %H:%M:%S'))
parse_dates = [data['timestamp']]

# Creating Weekday input
wd_input = np.array(data['timestamp'].map(lambda i: int(i.weekday())))

# Creating inputs sin\cos
seconds_in_day = 24*60*60
data_seconds = np.array(data['timestamp'].map(lambda i: i.weekday()))
input_sin = np.array(np.sin(2*np.pi*secondsperday/seconds_in_day))
input_cos = np.array(np.cos(2*np.pi*secondsperday/seconds_in_day))

# Putting inputs together in array
df = pd.DataFrame(data = {'value':data['value'], 'input_sin':input_sin, 'input_cos':input_cos, 'input_wd': wd_input})
column_indices = {name: i for i, name in enumerate(data.columns)}
n = len(df)
train_df = pd.DataFrame(df[0:int(n*0.7)])
val_df = pd.DataFrame(df[int(n*0.7):int(n*0.9)])
test_df = pd.DataFrame(df[int(n*0.9):])
num_features = df.shape[1]

# Standardization
train_mean = train_df['value'].mean()
train_std = train_df['value'].std()
train_df['value'] = (train_df['value'] - train_mean) / train_std
val_df['value'] = (val_df['value'] - train_mean) / train_std
test_df['value'] = (test_df['value'] - train_mean) / train_std

# 1st degree differencing
train_df['value'] = train_df['value'] - train_df['value'].shift()

# Handle negative values in 'value' for loging
train_df['value'] = train_df['value'].map(lambda i: abs(i))
train_df.loc[train_df.value <= 0, 'value'] = 0.000000001
train_df['value'] = train_df['value'].map(lambda i: np.log(i))
train_df = train_df.replace(np.nan, 0.000000001)

# 1st degree differencing
val_df['value'] = val_df['value'] - val_df['value'].shift()

# Handle negative values in 'value' for loging
val_df['value'] = val_df['value'].map(lambda i: abs(i))
val_df.loc[val_df.value <= 0, 'value'] = 0.000000001
val_df['value'] = val_df['value'].map(lambda i: np.log(i))
val_df = val_df.replace(np.nan, 0.000000001)

# 1st degree differencing
test_df['value'] = test_df['value'] - test_df['value'].shift()

# Handle negative values in 'value' for loging
test_df['value'] = test_df['value'].map(lambda i: abs(i))
test_df.loc[test_df.value <= 0, 'value'] = 0.000000001
test_df['value'] = test_df['value'].map(lambda i: np.log(i))
test_df = test_df.replace(np.nan, 0.000000001)

# Creating data window for forecast based on window size

class WindowGenerator():
  def __init__(self, input_width, label_width, shift,
               train_df=train_df, val_df=val_df, test_df=test_df,
               label_columns=None):
    # Store the raw data.
    self.train_df = train_df
    self.val_df = val_df
    self.test_df = test_df

    # Work out the label column indices.
    self.label_columns = label_columns
    if label_columns is not None:
      self.label_columns_indices = {name: i for i, name in
                                    enumerate(label_columns)}
    self.column_indices = {name: i for i, name in
                           enumerate(train_df.columns)}

    # Work out the window parameters.
    self.input_width = input_width
    self.label_width = label_width
    self.shift = shift

    self.total_window_size = input_width + shift

    self.input_slice = slice(0, input_width)
    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

    self.label_start = self.total_window_size - self.label_width
    self.labels_slice = slice(self.label_start, None)
    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

  def __repr__(self):
    return '\n'.join([
        f'Total window size: {self.total_window_size}',
        f'Input indices: {self.input_indices}',
        f'Label indices: {self.label_indices}',
        f'Label column name(s): {self.label_columns}'])

def split_window(self, features):
  inputs = features[:, self.input_slice, :]
  labels = features[:, self.labels_slice, :]
  if self.label_columns is not None:
    labels = tf.stack(
        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
        axis=-1)

  # Slicing doesn't preserve static shape information, so set the shapes
  # manually. This way the `tf.data.Datasets` are easier to inspect.
  inputs.set_shape([None, self.input_width, None])
  labels.set_shape([None, self.label_width, None])

  return inputs, labels

WindowGenerator.split_window = split_window

# Plotting function
def plot(self, model=None, plot_col='value', max_subplots=3):
  inputs, labels = self.example
  plt.figure(figsize=(12, 8))
  plot_col_index = self.column_indices[plot_col]
  max_n = min(max_subplots, len(inputs))
  for n in range(max_n):
    plt.subplot(3, 1, n+1)
    plt.ylabel(f'{plot_col} [normed]')
    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
             label='Inputs', marker='.', zorder=-10)
    if self.label_columns:
      label_col_index = self.label_columns_indices.get(plot_col, None)
    else:
      label_col_index = plot_col_index
    if label_col_index is None:
      continue
    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                edgecolors='k', label='Labels', c='#2ca02c', s=64)
    if model is not None:
      predictions = model(inputs)
      plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                  marker='X', edgecolors='k', label='Predictions',
                  c='#ff7f0e', s=64)
    if n == 0:
      plt.legend()
  plt.xlabel('Time [h]')
WindowGenerator.plot = plot

# Transforming data into tf dataset
def make_dataset(self, data):
  data = np.array(data, dtype=np.float64)
  ds = tf.keras.preprocessing.timeseries_dataset_from_array(
      data=data,
      targets=None,
      sequence_length=self.total_window_size,
      sequence_stride=1,
      shuffle=True,
      batch_size=32,)

  ds = ds.map(self.split_window)
  return ds
WindowGenerator.make_dataset = make_dataset

@property
def train(self):
  return self.make_dataset(self.train_df)

@property
def val(self):
  return self.make_dataset(self.val_df)

@property
def test(self):
  return self.make_dataset(self.test_df)

@property
def example(self):
  """Get and cache an example batch of `inputs, labels` for plotting."""
  result = getattr(self, '_example', None)
  if result is None:
    # No example batch was found, so get one from the `.train` dataset
    result = next(iter(self.train))
    # And cache it for next time
    self._example = result
  return result
WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example
single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['value'])

# Baseline model for comparison
class Baseline(tf.keras.Model):
  def __init__(self, label_index=None):
    super().__init__()
    self.label_index = label_index

  def call(self, inputs):
    if self.label_index is None:
      return inputs
    result = inputs[:, :, self.label_index]
    return result[:, :, tf.newaxis]

baseline = Baseline(label_index=column_indices['value'])
baseline.compile(loss=tf.losses.MeanSquaredError(),
                 metrics=[tf.metrics.MeanAbsoluteError()])
val_performance = {}
performance = {}
val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)
wide_window = WindowGenerator(
    input_width=25, label_width=25, shift=1,
    label_columns=['value'])
wide_window.plot(baseline)

# Function for compiling and fitting model and data
MAX_EPOCHS = 20
def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.SGD(),
                metrics=[tf.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

### LSTM ###
# Main Focus here is THIS model. Simple 2-layer LSTM for basic ts forecast.
lstm_model = tf.keras.models.Sequential([
    # Shape [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])
wide_window = WindowGenerator(
    input_width=50, label_width=50, shift=1,
    label_columns=['value'])
history = compile_and_fit(lstm_model, wide_window)
IPython.display.clear_output()
val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
wide_window.plot(lstm_model)
