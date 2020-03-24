# Instruments transformer

The goal of the model is to transpose a sound file of an instrument, to another instrument.
The initiation project focus on classical guitar and piano.

# Dataset
The dataset is gather from Youtube: around 10h of music, cut in 1s lenght wav sample.

# Pre-processing



# Model
The structure chosen is the LSTM auto-encoder (See articles below for further details).
The global structure is:
Encoder:  1. LSTM layer (N units + param1)
          2. LSTM layer (M units + param2)
          
Decoder:  3. RepeatVector layer
          4. LSTM layer (M units + param2)
          5. LSTM layer (N units + param1)
          6. TimeDistributed Dense layer

The RepeatVector layer is need to get the correct shape for the decoder
The Time Distributed layer permit to get our initial structure.


# Source of interest

Links to article of interests in this project:
- https://machinelearningmastery.com/lstm-autoencoders/
https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352

