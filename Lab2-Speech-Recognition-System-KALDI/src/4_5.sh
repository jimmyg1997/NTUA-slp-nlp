. ./path.sh # Needed for KALDI_ROOT
. ./cmd.sh
#===================================4_5_1====================================
cd ~/kaldi/egs/wsj/s5

# ------------------- Data preparation for DNN -------------------- #
# Compute cmvn stats for every set and save them in specific .ark files
# These will be used by the python dataset class that you were given
#for set in train dev test; do
#  compute-cmvn-stats --spk2utt=ark:data/${set}/spk2utt scp:data/${set}/feats.scp ark:data/${set}/${set}"_cmvn_speaker.ark"
#  compute-cmvn-stats scp:data/${set}/feats.scp ark:data/${set}/${set}"_cmvn_snt.ark"
#done

#compute-cmvn-stats --spk2utt=ark:data/train_1kshort/spk2utt scp:data/train_1kshort/feats.scp ark:data/train_1kshort/train_1kshort"_cmvn_speaker.ark"
#compute-cmvn-stats scp:data/train_1kshort/feats.scp ark:data/train_1kshort/train_1kshort"_cmvn_snt.ark"

#steps/align_si.sh --nj 3 --cmd "$train_cmd" \
#  data/train_1kshort data/lang_test  exp/tri1 exp/tri1_ali

#steps/align_si.sh --nj 3 --cmd "$train_cmd" \
#  data/dev data/lang_test exp/tri1 exp/tri1_dev_ali 

#steps/align_si.sh --nj 3 --cmd "$train_cmd" \
#	data/test data/lang_test exp/tri1 exp/tri1_test_ali

#===================================4_5_2====================================

#copy-feats scp:/Users/jimmyg1997/kaldi/egs/wsj/s5/data/dev/feats.scp ark:- | 
#apply-cmvn --utt2spk=ark:/Users/jimmyg1997/kaldi/egs/wsj/s5/data/dev/utt2spk 
#ark:/Users/jimmyg1997/kaldi/egs/wsj/s5/data/dev/dev_cmvn_speaker.ark ark:- ark:- | 
#add-deltas --delta-order=2 ark:- ark:- | splice-feats --left-context=3 --right-context=3 ark:- ark:feats

#gunzip -c /Users/jimmyg1997/kaldi/egs/wsj/s5/exp/tri1_dev_ali/ali.*.gz | ali-to-pdf 
#/Users/jimmyg1997/kaldi/egs/wsj/s5/exp/tri1_dev_ali/final.mdl ark:- ark:labels