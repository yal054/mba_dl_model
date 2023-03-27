#!/usr/bin/env python
# Copyright 2017 Calico LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import print_function
from optparse import OptionParser
import json
import os
import pdb
import sys
import time

import h5py
from intervaltree import IntervalTree
import joblib
import numpy as np
import pandas as pd
from scipy.stats import poisson
from scipy.stats import pearsonr

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve, average_precision_score
import tensorflow as tf

import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
import seaborn as sns

from basenji import bed
from basenji import dataset
from basenji import plots
from basenji import seqnn
from basenji import trainer

if tf.__version__[0] == '1':
  tf.compat.v1.enable_eager_execution()

"""
basenji_test_bdg.py

Test the accuracy of a trained model.
"""

################################################################################
# main
################################################################################
def main():
  usage = 'usage: %prog [options] <targets_bdg> <preds_bdg>'
  parser = OptionParser(usage)
  parser.add_option('-p', dest='prefix',
      default='test',
      help='Output prefix [Default: %default]')
  parser.add_option('-o', dest='out_dir',
      default='test_out',
      help='Output directory for test statistics [Default: %default]')
  (options, args) = parser.parse_args()

  if len(args) != 2:
    parser.error('Must provide parameters, model, and test data HDF5')
  else:
    targets_bdg = args[0]
    preds_bdg = args[1]

  if not os.path.isdir(options.out_dir):
    os.mkdir(options.out_dir)
  
  outprfx = options.prefix

  #######################################################

  # read targets
  targets_df = pd.read_csv(targets_bdg, sep='\t', header=None)
  targets_df.columns =['chr', 'start', 'end', 'targets']
  
  # read preds
  preds_df = pd.read_csv(preds_bdg, sep='\t', header=None)
  preds_df.columns =['chr', 'start', 'end', 'preds']

  
  #######################################################
  # extract signals

  test_targets_ti_flat = targets_df['targets']
  test_preds_ti_flat = preds_df['preds']


  ############################################
  # scatter

  # take log2
  test_targets_ti_log = np.log2(test_targets_ti_flat + 1)
  test_preds_ti_log = np.log2(test_preds_ti_flat + 1)

  r = pearsonr(test_targets_ti_log, test_preds_ti_log)[0]

  # plot log2
  sns.set(font_scale=1.2, style='ticks')
  out_pdf = '%s/%s.scatter.pdf' % (options.out_dir, outprfx)
  plots.regplot(
          test_targets_ti_log,
          test_preds_ti_log,
          out_pdf,
          poly_order=1,
          alpha=0.3,
          sample=500,
          figsize=(6, 6),
          x_label='log2 Experiment',
          y_label='log2 Prediction',
          table=True)
        
    
  ############################################
  # call peaks
  
  test_targets_ti_lambda = np.mean(test_targets_ti_flat)
  test_targets_pvals = 1 - poisson.cdf(
          np.round(test_targets_ti_flat) - 1, mu=test_targets_ti_lambda)
  test_targets_qvals = np.array(ben_hoch(test_targets_pvals))
  test_targets_peaks = test_targets_qvals < 0.01
  test_targets_peaks_str = np.where(test_targets_peaks, 'Peak',
                                        'Background')

  ############################################
  # ROC
      
  plt.figure()
  fpr, tpr, _ = roc_curve(test_targets_peaks, test_preds_ti_flat)
  auroc = roc_auc_score(test_targets_peaks, test_preds_ti_flat)
  
  plt.plot([0, 1], [0, 1], c='black', linewidth=1, linestyle='--', alpha=0.7)
  plt.plot(fpr, tpr, c='black')
  ax = plt.gca()
  ax.set_xlabel('False positive rate')
  ax.set_ylabel('True positive rate')
  ax.text(
          0.99, 0.02, 'AUROC %.3f' % auroc,
          horizontalalignment='right')  # , fontsize=14)
  ax.grid(True, linestyle=':')
  
  plt.savefig('%s/%s.roc.pdf' % (options.out_dir, outprfx))
  plt.close()

  ############################################
  # PR
    
  plt.figure()
  prec, recall, _ = precision_recall_curve(test_targets_peaks,
                                               test_preds_ti_flat)
  auprc = average_precision_score(test_targets_peaks, test_preds_ti_flat)

  plt.axhline(
          y=test_targets_peaks.mean(),
          c='black',
          linewidth=1,
          linestyle='--',
          alpha=0.7)
  plt.plot(recall, prec, c='black')
  ax = plt.gca()
  ax.set_xlabel('Recall')
  ax.set_ylabel('Precision')
  ax.text(
          0.99, 0.95, 'AUPRC %.3f' % auprc,
          horizontalalignment='right')  # , fontsize=14)
  ax.grid(True, linestyle=':')
    
  plt.savefig('%s/%s.pr.pdf' % (options.out_dir, outprfx))
  plt.close()

  ############################################
  # statistic
  
  sta_out = [outprfx, str(r), str(auroc), str(auprc)]
  
  out_file = '%s/%s.sta.txt' % (options.out_dir, outprfx)
  f = open(out_file, "w") #open a file in write mode
  f.write("\t".join(sta_out) + '\n') #write the tuple into a file
  f.close() 

def ben_hoch(p_values):
  """ Convert the given p-values to q-values using Benjamini-Hochberg FDR. """
  m = len(p_values)

  # attach original indexes to p-values
  p_k = [(p_values[k], k) for k in range(m)]

  # sort by p-value
  p_k.sort()

  # compute q-value and attach original index to front
  k_q = [(p_k[i][1], p_k[i][0] * m // (i + 1)) for i in range(m)]

  # re-sort by original index
  k_q.sort()

  # drop original indexes
  q_values = [k_q[k][1] for k in range(m)]

  return q_values


################################################################################
# __main__
################################################################################
if __name__ == '__main__':
  main()
