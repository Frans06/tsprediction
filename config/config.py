# -*- coding: utf-8 -*-
"""
This is a config file por enviroment variable definitions

Example:
    import and inherit Class::

        import config

Class Config define global and enviromental Variables.

Todo:
    * For module TODOs
    * You have to also use ``sphinx.ext.todo`` extension
"""

class Config: # pylint: disable-msg=R0903
    '''
        Simple config class to be heritated
    '''
    APP_NAME = 'myapp'
    SECRET_KEY = 'secret-key-of-myapp'
    ADMIN_NAME = 'administrator'

    AWS_DEFAULT_REGION = 'ap-northeast-2'

    STATIC_PREFIX_PATH = 'static'
    ALLOWED_IMAGE_FORMATS = ['jpg', 'jpeg', 'png', 'gif']
    MAX_IMAGE_SIZE = 5242880 # 5MB
    DEBUG = False
    TESTING = False
    ENV = None

    def __init__(self, LOGGER, enviroment):
        LOGGER.info('Defining Enviroment variables:: App name: %s', self.APP_NAME)
        LOGGER.info('   Enviroment: %s', self.ENV)
        LOGGER.info('   Debug: %s', self.DEBUG)
        LOGGER.info('   Testing: %s', self.TESTING)
        self.enviroment = enviroment
class DevelopmentConfig(Config): # pylint: disable-msg=R0903
    '''
    Config development case
    '''
    DEBUG = True
    ENV = 'DEV'

    AWS_ACCESS_KEY_ID = 'aws-access-key-for-dev'
    AWS_SECERT_ACCESS_KEY = 'aws-secret-access-key-for-dev'
    AWS_S3_BUCKET_NAME = 'aws-s3-bucket-name-for-dev'

    DATABASE_URI = 'database-uri-for-dev'

class TestConfig(Config): # pylint: disable-msg=R0903
    '''
    Config testing case
    '''
    DEBUG = True
    TESTING = True
    ENV = 'TEST'

    AWS_ACCESS_KEY_ID = 'aws-access-key-for-test'
    AWS_SECERT_ACCESS_KEY = 'aws-secret-access-key-for-test'
    AWS_S3_BUCKET_NAME = 'aws-s3-bucket-name-for-test'

    DATABASE_URI = 'database-uri-for-dev'


class ProductionConfig(Config): # pylint: disable-msg=R0903
    '''
    Config Production case
    '''
    DEBUG = False
    ENV = 'PROD'

    AWS_ACCESS_KEY_ID = 'aws-access-key-for-prod'
    AWS_SECERT_ACCESS_KEY = 'aws-secret-access-key-for-prod'
    AWS_S3_BUCKET_NAME = 'aws-s3-bucket-name-for-prod'

    DATABASE_URI = 'database-uri-for-dev'


class CIConfig: # pylint: disable-msg=R0903
    '''
    Continue Integration config
    '''
    SERVICE = 'travis-ci'
    HOOK_URL = 'web-hooking-url-from-ci-service'
