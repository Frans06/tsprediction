"""
This is a plot module for enviroment variable definitions

Example:
    import and inherit Class::

        import plots

Class Config define global and enviromental Variables.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension
"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
class Data():
    '''
    Create or dowload some data for test or train
    '''
    SEQ_LEN = 10
    def __int__(self):
        self.to_csv('train.csv', 1000)  # 1000 sequences
        self.to_csv('valid.csv', 50)

    def create_time_series(self):
        '''
        create a random time series as signal data
        '''
        freq = (np.random.random()*0.5) + 0.1
        ampl = np.random.random() + 0.5  # 0.5 to 1.5
        x_axis = np.sin(np.arange(0, self.SEQ_LEN) * freq) * ampl
        return x_axis

    def to_csv(self, filename, sequences):
        '''
        write data to csv
        '''
        with open(filename, 'w') as ofp:
            for line in range(0, sequences):
                seq = self.create_time_series()
                line = ",".join(map(str, seq))
                ofp.write(line + '\n')

    def plot(self):
        '''
        plotting generated data
        '''
        for _ in range(0, 5):
            sns.tsplot(self.create_time_series())# 5 series

if __name__ == "__main__":
    DATA = Data()
    DATA.plot()
    plt.show()
