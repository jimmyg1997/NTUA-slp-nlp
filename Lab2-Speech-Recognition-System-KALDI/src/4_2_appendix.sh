
. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

cd "data/local/lm_tmp"
export IRSTLM=$KALDI_ROOT/tools/irstlm/
export PATH=${PATH}:$IRSTLM/bin


echo "########################################################"
echo "Calculating perplexity for evaluation data unigram model"
echo "########################################################"
prune-lm --threshold=1e-6,1e-6 lm_dev1.ilm.gz lm_dev1.plm
compile-lm lm_dev1.lm --eval=../dict/lm_dev.text --dub=10000000

echo "#######################################################"
echo "Calculating perplexity for evaluation data bigram model"
echo "#######################################################"
prune-lm --threshold=1e-6,1e-6 lm_dev2.ilm.gz lm_dev2.plm
compile-lm lm_dev2.lm --eval=../dict/lm_dev.text --dub=10000000

echo "##################################################"
echo "Calculating perplexity for test data unigram model"
echo "##################################################"
prune-lm --threshold=1e-6,1e-6 lm_test1.ilm.gz lm_test1.plm
compile-lm lm_test1.lm --eval=../dict/lm_test.text --dub=10000000

echo "#################################################"
echo "Calculating perplexity for test data bigram model"
echo "#################################################"
prune-lm --threshold=1e-6,1e-6 lm_test2.ilm.gz lm_test2.plm
compile-lm lm_test2.lm --eval=../dict/lm_test.text --dub=10000000