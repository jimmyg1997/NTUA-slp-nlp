
. ./path.sh # Needed for KALDI_ROOT

export IRSTLM=$KALDI_ROOT/tools/irstlm/
export PATH=${PATH}:$IRSTLM/bin

cd "data/local/dict"

build-lm.sh -i "lm_train.text" -n 1 -o lm_train1.ilm.gz
build-lm.sh -i "lm_train.text" -n 2 -o lm_train2.ilm.gz
build-lm.sh -i "lm_test.text" -n 1 -o lm_test1.ilm.gz
build-lm.sh -i "lm_test.text" -n 2 -o lm_test2.ilm.gz
build-lm.sh -i "lm_dev.text" -n 1 -o lm_dev1.ilm.gz
build-lm.sh -i "lm_dev.text" -n 2 -o lm_dev2.ilm.gz

mv lm_train1.ilm.gz ../lm_tmp
mv lm_train2.ilm.gz ../lm_tmp
mv lm_test1.ilm.gz ../lm_tmp
mv lm_test2.ilm.gz ../lm_tmp
mv lm_dev1.ilm.gz ../lm_tmp
mv lm_dev2.ilm.gz ../lm_tmp

cd "../../../"