SFO Weather Forecaster Prototype - technical specifications


Input:

Past history window of 5 days, at 3 hour intervals;

6 feature columns: tmpc (temprature in Celsius),
                   dwpc (dew point in Celsius),
                   relh (relative humidity in %),
                   sped (wind speed in mph),
                   alti (altimeter in inches (indicate air pressure)),
                   vsby (visibility in miles)


Output:

Future target window of 24 hours

1 feature: tmpc


Model architecture:

LSTM(128), Dropout, LSTM(64), Dropout, LSTM(32), Dropout, Dense(128), Dropout, Dense(64), Dropout, Dense(24)

Optimizer: Adam optimization algorithm

Learning rate: optimal = 0.008, default = 0.001

Loss function: mae (mean absolute error)


Training data:

623,725 samples * 6 feature columns

500,000 samples for training, the rest for validation


Current loss after training:

mae = ~ 0.35 (after input normalization)
