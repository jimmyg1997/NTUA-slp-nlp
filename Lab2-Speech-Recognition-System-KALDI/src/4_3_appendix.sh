. ./path.sh # Needed for KALDI_ROOT
. ./cmd.sh
#===================================4_5_1====================================
cd ~/kaldi/egs/wsj/s5
#This is for all the MFCC features.
feat-to-dim ark:mfcc_test/raw_mfcc_test.1.ark -

#Number of frames
feat-to-len scp:data/test/feats.scp ark,t:data/test/feats.lengths
cat data/test/feats.lengths


