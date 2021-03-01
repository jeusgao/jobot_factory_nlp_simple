import os
import tensorflow as tf
from distutils.util import strtobool

__all__ = [
    'keras', 'TF_KERAS', 'V_TF',
]

TF_KERAS = strtobool(os.environ.get('TF_KERAS', '0'))
V_TF = tf.__version__

if TF_KERAS or V_TF >= '2.3':
    import tensorflow as tf
    import tensorflow.keras.backend as K
    keras = tf.keras
else:
    import keras
    import keras.backend as K
