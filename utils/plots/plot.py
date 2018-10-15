# -*- coding: utf-8 -*-
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

import seaborn as sns

class PlotInputData(): # pylint: disable-msg=R0903
    '''
    Simple input data plotter
    '''
    def __init__(self, datainput):
        for _ in range(0, 5):
            sns.tsplot(datainput)

    @staticmethod
    def other_plotters():
        '''
        other ploting methods
        '''
        print('plotting hard')
