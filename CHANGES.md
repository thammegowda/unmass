
# Changes:

1. python module. `pip install` -able
1. Train without peeking into held out `test` set. Use test set when all the training is finished 
1. Runs on CPU too (for development/debugging on laptops that dont have GPUs)
1. Generalized preprocessing script that can be used with any two languages, just point the local path
1. Using `sacrebleu` instead of perl script (that actually came without executable permission)
   * Also, sacremoses normalizer and tokenizer instead of moses perl scripts 
1. make this work on pytorch 1.5 or newer, so we can use newer cuda
