import tensorflow as tf 
import numpy as np
import seaborn as sns
import pandas as pd
import shutil
from tensorflow.contrib.learn import ModeKeys
import tensorflow.contrib.rnn as rnn

class Data():
    SEQ_LEN = 10 
    def __int__(self):
        to_csv('train.csv', 1000)  # 1000 sequences
        to_csv('valid.csv',  50)

    def create_time_series():
        freq = (np.random.random()*0.5) + 0.1
        ampl = np.random.random() + 0.5  # 0.5 to 1.5
        x = np.sin(np.arange(0,SEQ_LEN) * freq) * ampl
        return x
    
    def to_csv(filename, N):
      with open(filename, 'w') as ofp:
        for lineno in range(0, N):
          seq = create_time_series()
          line = ",".join(map(str, seq))
          ofp.write(line + '\n')

