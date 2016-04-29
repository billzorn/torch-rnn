th train.lua

-unk 

-model_type   lstm
-wordvec_size 64
-rnn_size     128
-num_layers   2
-dropout      0.0
-batchnorm    0

-grad_clip        5
-learning_rate    0.002
-lr_decay_factor  0.5
-lr_decay_batches 1000
-lr_decay_epochs  5

-batch_size 50
-seq_length 50

-max_batches 10000
-max_epochs  50

-print_every      1
-eval_val_every   1000
-checkpoint_every 1000
