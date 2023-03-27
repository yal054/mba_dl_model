"""Weighted Poisson Loss."""

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils import losses_utils
from tensorflow.python.keras.losses import LossFunctionWrapper

################################################################################
# Losses
################################################################################

def poisson_dotcov(y_true, y_pred, cov_weight=1, epsilon=1e-6):
  type_len = y_true.shape[2]
  seq_len = y_true.shape[1]
    
  # poisson loss
  poisson_loss = y_pred - y_true * tf.math.log(y_pred)

  # add epsilon to protect against all tiny values
  y_true += epsilon
  y_pred += epsilon

  # normalize to sum to one
  yn_true = y_true / tf.math.reduce_sum(y_true, axis=-2, keepdims=True)

  # dot cov
  covmx = tfp.stats.covariance(y_true, sample_axis=1, event_axis=-1)
  cov_sum = tf.reduce_sum(covmx, axis=-1)
  cov_norm = cov_sum / tf.reduce_sum(cov_sum) 
  cov_norm = tf.cast(cov_norm, tf.float32)
  scale_vec = tf.repeat(1/type_len, type_len)
  weight_vec = cov_norm / tf.add(scale_vec, cov_norm) / 2 
  weight_vec = tf.expand_dims(weight_vec, axis=1)
  #weight_vec = tf.cast(weight_vec, tf.float64)

  # weighted combination
  cov_loss = tf.reduce_mean(poisson_loss + cov_weight*weight_vec, -1)
    
  return cov_loss

class PoissonDOTCOV(LossFunctionWrapper):
  def __init__(self, cov_weight=1, reduction=losses_utils.ReductionV2.AUTO, name='poisson_dotcov'):
    self.cov_weight = cov_weight
    pois_dotcov = lambda yt, yp: poisson_dotcov(yt, yp, self.cov_weight)
    super(PoissonDOTCOV, self).__init__(
        pois_dotcov, name=name, reduction=reduction)

