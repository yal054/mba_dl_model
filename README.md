# mba_dl_model

Mouse brain atlas deep learning model.

Current model was trained on snATAC-seq using [basenji](https://github.com/calico/basenji).

- **model_best.h5:** The best model for 343 cell types
- **params.json:** Parameters used for training
- **weighted_poisson_loss.py:** weighted poisson loss function
- **basenji_test_bdg.py:** script for testing performance between two bedgraph files.
