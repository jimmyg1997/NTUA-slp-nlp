. ./path.sh # Needed for KALDI_ROOT
. ./cmd.sh
#===================================4_5_1====================================
cd ~/kaldi/egs/wsj/s5

feat-to-dim ark:mfcc/raw_mfcc_train.1.ark -
