# General study on music
The goal of this repository is to study different model of deep learning in the musical area.
It focus on two subjects: the classification of piano composer, and the classification of music instruments.
The analysis were done in Python 3, using Google Colab and Google Cloud Plateform.

# Dataset
Two datasets were used for this project:
The first was create from youtube videos of five piano composers (using youtube-dl library).
The second came from the website https://www.upf.edu/web/mtg/irmas, and came with an article on instrumant recognition.

# Pre-processing
Preprocessing is a bit tricky, as it concerns audio files. Audio can have multiple representation, some of them can be found in the notebook "data_vizualization". In the litterature, the Fourier Short Term Transform (https://en.wikipedia.org/wiki/Short-time_Fourier_transform, STFT) is the most used for this kind of analysis. It is a succession of Fourier Transform on small time windows to get the successives frequencies distributions in this time windows. The most important parameter is the width of this time window, as it impacts the time precision, and the frequency precision (see more details in the link above).
Librosa library give the possibility to those transformation.

Two way of storage were tested, jpg and npy. Jpg give a compressed image of the Fourier Transform, which reduce the size of storage needed. Files was small enough to be used on Google Collaboratory Notebook. However the compression implies a loss of information. Therefore the .npy object, which are basically the storage of the numpy array representation of the STFT is choosen. I used Google Cloud Platform for running the deep learning model.



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

