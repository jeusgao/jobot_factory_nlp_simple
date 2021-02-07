import os
from distutils.util import strtobool
import tensorflow as tf

__all__ = [
    'keras', 'TF_KERAS', 'V_TF',
]

TF_KERAS = strtobool(os.environ.get('TF_KERAS', '0'))
V_TF = float('.'.join(tf.__version__.split('.')[:2]))

if TF_KERAS or V_TF >= 2.3:
    import tensorflow as tf
    keras = tf.keras
else:
    import keras
