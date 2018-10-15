# -*- coding: utf-8 -*-
"""

Time series predictions based on LSTM Cells using TF or Keras. Simple
implementation and consult api for delploy

Example:
    In order to run program we need to install pipenv and run it after install
    dependencies

        $ python main.py

Attributes:
    Add modules later
Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import shutil
import tensorflow as tf
from tensorflow.contrib.learn import ModeKeys # pylint: disable-msg=E0611
import tensorflow.contrib.rnn as rnn # pylint: disable-msg=E0611,C0414
from colorlog import ColoredFormatter

import config
import utils.plots

class App():
    '''
    App class. General Class initialization
    '''
    SEQ_LEN = 10
    DEFAULTS = [[0.0] for x in range(0, SEQ_LEN)]
    BATCH_SIZE = 20
    TIMESERIES_COL = 'rawdata'
    N_OUTPUTS = 2  # in each sequence, 1-8 are features, and 9-10 is label
    N_INPUTS = SEQ_LEN - N_OUTPUTS
    LSTM_SIZE = 3
    def __init__(self):
        self.config = None
        self.ci_config = None
    @classmethod
    def read_dataset(cls, filename, mode=ModeKeys.TRAIN):
        """
        Reading CSV dataset
        """
        def _input_fn():
            num_epochs = 100 if mode == ModeKeys.TRAIN else 1

        # could be a path to one file or a file pattern.
            input_file_names = tf.train.match_filenames_once(filename)


            filename_queue = tf.train.string_input_producer(
                input_file_names, num_epochs=num_epochs, shuffle=True)
            reader = tf.TextLineReader()
            _, value = reader.read_up_to(filename_queue, num_records=cls.BATCH_SIZE)

            value_column = tf.expand_dims(value, -1, name='value')
            logging.debug('readcsv=%s', value_column)

            # all_data is a list of tensors
            all_data = tf.decode_csv(value_column, record_defaults=cls.DEFAULTS)
            inputs = all_data[:len(all_data)-cls.N_OUTPUTS]  # first few values
            label = all_data[len(all_data)-cls.N_OUTPUTS : ] # last few values

            # from list of tensors to tensor with one more dimension
            inputs = tf.concat(inputs, axis=1)
            label = tf.concat(label, axis=1)
            logging.debug('inputs=%s', inputs)
            return {cls.TIMESERIES_COL: inputs}, label   # dict of features, label

        return _input_fn

    def simple_rnn(self, features, labels, mode):
        """
        Defining simple resurrent Neuronal Network
        """
        # 0. Reformat input shape to become a sequence
        x_axis = tf.split(features[self.TIMESERIES_COL], self.N_INPUTS, 1)
        #print 'x={}'.format(x)

        # 1. configure the RNN
        lstm_cell = rnn.BasicLSTMCell(self.LSTM_SIZE, forget_bias=1.0)
        outputs, _ = tf.nn.static_rnn(lstm_cell, x_axis, dtype=tf.float32)
        # slice to keep only the last cell of the RNN

        outputs = outputs[-1]
        #print 'last outputs={}'.format(outputs)

        # output is result of linear activation of last layer of RNN
        weight = tf.Variable(tf.random_normal([self.LSTM_SIZE, self.N_OUTPUTS]))
        bias = tf.Variable(tf.random_normal([self.N_OUTPUTS]))
        predictions = tf.matmul(outputs, weight) + bias

        # 2. loss function, training/eval ops
        if mode in (ModeKeys.TRAIN, ModeKeys.EVAL):
            loss = tf.losses.mean_squared_error(labels, predictions)
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=0.01,
                optimizer="SGD")
            eval_metric_ops = {
                "rmse": tf.metrics.root_mean_squared_error(labels, predictions)
            }
        else:
            loss = None
            train_op = None
            eval_metric_ops = None

        # 3. return ModelFnOps
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions={"predicted": predictions},
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            export_outputs={"predicted": tf.estimator.export.PredictOutput(predictions)})

    def get_train(self):
        '''
        training model method
        '''
        return self.read_dataset('train.csv', mode=ModeKeys.TRAIN)

    def get_valid(self):
        '''
        validate model method
        '''
        return self.read_dataset('valid.csv', mode=ModeKeys.EVAL)

    def serving_input_receiver_fn(self):
        '''
        serving data to estimator
        '''
        feature_placeholders = {
            self.TIMESERIES_COL: tf.placeholder(tf.float32, [None, self.N_INPUTS])
        }
        features = {
            key: tf.expand_dims(tensor, -1)
            for key, tensor in feature_placeholders.items()
        }
        features[self.TIMESERIES_COL] = tf.squeeze(features[self.TIMESERIES_COL],
                                                   axis=[2], name='timeseries')
        logging.debug('serving: features=%s', features[self.TIMESERIES_COL])
        return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

    def experiment_fn(self, output_dir):
        '''
        main method to train model
        '''
        train_spec = tf.estimator.TrainSpec(input_fn=self.get_train(), max_steps=1000)
        exporter = tf.estimator.FinalExporter('timeseries',
                                              self.serving_input_receiver_fn)
        eval_spec = tf.estimator.EvalSpec(input_fn=self.get_valid(),
                                          exporters=[exporter])
        estimator = tf.estimator.Estimator(model_fn=self.simple_rnn, model_dir=output_dir)
        tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


def setup_logger(name=__name__, propagation=True):
    """Return a logger with a default ColoredFormatter."""
    # TODO: Add Log File option
    formatter = ColoredFormatter(
        "%(log_color)s%(levelname)-8s::%(name)-12s=> %(reset)s %(blue)s%(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red',
        }
    )

    logger = logging.getLogger(name)
    for handlers in logger.handlers:
        logger.removeHandler(handlers)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = propagation
    logger = set_logger_level(logger)
    return logger

def set_logger_level(logger):
    """Setting logger level acconrding to env variable"""
    if os.getenv('LOG') != 'TRUE':
        if os.getenv('ENV') == 'DEV':
            logger.setLevel(level=logging.DEBUG)
        elif os.getenv('ENV') == 'PROD':
            logger.setLevel(level=logging.WARN)
        else:
            logger.setLevel(level=logging.INFO)
    return logger

if __name__ == '__main__':

    ENV = os.getenv('ENV', None)
    TF_LOGGER = setup_logger('tensorflow', False)
    LOGGER = setup_logger()
    APP = App()
    if ENV is None:
        LOGGER.info("Environment variable 'ENV' not set, returning development configs.")
        ENV = 'DEV'
    if ENV == 'DEV':
        APP.config = config.DevelopmentConfig(LOGGER, ENV)
    elif ENV == 'TEST':
        APP.config = config.TestConfig(LOGGER, ENV)
    elif ENV == 'PROD':
        APP.config = config.ProductionConfig(LOGGER, ENV)
    else:
        raise ValueError('Invalid environment name')
    APP.ci_config = config.CIConfig
    OUTPUT_DIR = '/home/frans/Documents/tsprediction/model'
    LOGGER.info('Cleanning ouput directory')
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True) # start fresh each time

    LOGGER.info('Outpout directory clean')
    APP.experiment_fn(OUTPUT_DIR)
