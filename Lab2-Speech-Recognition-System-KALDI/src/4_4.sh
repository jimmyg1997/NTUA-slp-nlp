. ./path.sh # Needed for KALDI_ROOT
. ./cmd.sh
#===================================Make the training====================================
# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesnt always help. [its to discourage non-silence
# models from modeling silence.]
#===================================4_4_1====================================
cd ~/kaldi/egs/wsj/s5

#steps/train_mono.sh --boost-silence 1.25 --nj 4 --cmd "$train_cmd" \
#  data/train data/lang_test exp/mono0a || exit 1;
#steps/align_si.sh --boost-silence 1.25 --nj 4 --cmd "$train_cmd" \
#  data/train data/lang_test exp/mono0a exp/mono0a_ali 

#===================================4_4_2====================================
#This script creates a fully expanded decoding graph (HCLG) that represents the 
#language-model, pronunciation dictionary (lexicon), context-dependency, and 
#HMM structure in our model. The output is a Finite State Transducer that has 
#word-ids on the output, and pdf-ids on the input (these are indexes that 
#resolve to Gaussian Mixture Models).
#Rename the G_x.fst we want to transform into HCLG.fst, to G.fst so that the 
#script 'mkgraph.sh' will understand it

#utils/mkgraph.sh --mono data/lang_test \
#   exp/mono0a exp/mono0a/graph_nosp_tgpr


#Specifically for the HLGC.fst for the bigram model we get the following:
#while [ ! -f data/lang_test/tmp/LG.fst ] || \
#   [ -z data/lang_test/tmp/LG.fst ]; do
#  sleep 20;
#done
#sleep 30


#===================================4_4_3====================================
#=============Make the testing with the trained model========================
#Begin configuration section.
#Decoding#
#--scoring-opts "--min-lmwt 1 --max-lmwt 20"
cmd=run.pl


steps/decode.sh --nj 3 --cmd "$decode_cmd" exp/mono0a/graph_nosp_tgpr \
   data/dev exp/mono0a/decode_dev || exit 1;
steps/decode.sh --nj 3 --cmd "$decode_cmd"  exp/mono0a/graph_nosp_tgpr \
   data/test exp/mono0a/decode_test || exit 1;


#===================================4_4_4=======================================
#======================Printing the best PER Result=============================
#Result is about dev/test for both models (1448, 1k utterneces)

[ -d exp/mono0a/decode_dev ] && grep WER exp/mono0a/decode_dev/wer_* | utils/best_wer.sh
[ -d exp/mono0a/decode_test ] && grep WER exp/mono0a/decode_test/wer_* | utils/best_wer.sh
#for x in exp/{mono,tri,sgmm,dnn,combine}*/decode*; do [ -d $x ] && echo $x | grep "${1:-.*}" >/dev/null && grep WER $x/wer_* 2>/dev/null | utils/best_wer.sh; done


#===================================4_4_5=======================================
#============================Make the division================================

#Now make subset with the shortest 1k utterances from train data.
#utils/subset_data_dir.sh --shortest data/train 1000 data/train_1kshort || exit 1;
#utils/data/fix_data_dir.sh data/train_1kshort


#Computes training alignments using a model with  delta+delta-delta features features. Specifically 
#Training the model of the subset of 1k utterances

#steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
# 	data/train_1kshort data/lang_test exp/mono0a_ali exp/tri1 || exit 1;

#steps/align_si.sh --boost-silence 1.25 --nj 4 --cmd "$train_cmd" \
#	data/train_1kshort data/lang_test exp/tri1 exp/tri1_ali || exit 1;

#NumLeaves = 2000: it can recognize all phone sequences, the numleaves determines 
#how many classes it splits them into for purposes of pooling similar 
#contexts.  It's analogous to (but not quite the same as) the number of 
#'physical' triphones in HTK.
#NumGauss = 10000: The number of Gaussians is the total over all leaves, so the average 
#num-gauss per pdf/leaf is num-gauss/num-leaves, but leaves with more 
#data will get somewhat more Gaussians. 

#=============Make AGAIN the testing with the new trained model=================
#Decoding again dev, test with the new trained model
#FEATURE TYPE IS DELTA

#utils/mkgraph.sh data/lang_test \
# exp/tri1 exp/tri1/graph_nosp_tgpr || exit 1;


#steps/decode.sh --nj 3 --cmd "$decode_cmd" exp/tri1/graph_nosp_tgpr \
#  data/dev exp/tri1/decode_dev_1kshort && \
#steps/decode.sh --nj 3 --cmd "$decode_cmd" exp/tri1/graph_nosp_tgpr \
#  data/test exp/tri1/decode_test_1kshort  || exit 1;


#compute-wer --text --mode=present ark:exp/tri1/decode_test_1kshort/scoring_kaldi/test_filt.txt ark,p:-
#compute-wer --text --mode=present ark:exp/tri1/decode_dev_1kshort/scoring_kaldi/dev_filt.txt ark,p:-

[ -d exp/tri1/decode_dev_1kshort ] && grep WER exp/tri1/decode_dev_1kshort/wer_* | utils/best_wer.sh
[ -d exp/tri1/decode_test_1kshort ] && grep WER exp/tri1/decode_test_1kshort/wer_* | utils/best_wer.sh


