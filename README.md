# General study on music
The goal of this repository is to study different model of deep learning in the musical area.
It focus on two subjects: the classification of piano composer, and the classification of music instruments.

# Dataset
Two datasets were used for this project:
The first was create from youtube videos of five piano composers.
The second came from the website https://www.upf.edu/web/mtg/irmas, and came with an article on instrumant recognition.

# Pre-processing
Preprocessing is a bit tricky, as it concerns audio files. Audio can have multiple representation, 


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

