

. ./path.sh # Needed for KALDI_ROOT

export IRSTLM=$KALDI_ROOT/tools/irstlm/
export PATH=${PATH}:$IRSTLM/bin


cd "data/local/lm_tmp"
compile-lm lm_train1.ilm.gz --text=yes lm_train1.lm
compile-lm lm_train1.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > lm_train1.arpa.gz

compile-lm lm_train2.ilm.gz --text=yes lm_train2.lm
compile-lm lm_train2.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > lm_train2.arpa.gz

compile-lm lm_test1.ilm.gz --text=yes lm_test1.lm
compile-lm lm_test1.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > lm_test1.arpa.gz

compile-lm lm_test2.ilm.gz --text=yes lm_test2.lm
compile-lm lm_test2.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > lm_test2.arpa.gz

compile-lm lm_dev1.ilm.gz --text=yes lm_dev1.lm
compile-lm lm_dev1.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > lm_dev1.arpa.gz

compile-lm lm_dev2.ilm.gz --text=yes lm_dev2.lm
compile-lm lm_dev2.ilm.gz -t=yes /dev/stdout | grep -v unk | gzip -c > lm_dev2.arpa.gz



mv lm_train1.arpa.gz ../nist_lm
mv lm_train2.arpa.gz ../nist_lm
mv lm_test1.arpa.gz ../nist_lm
mv lm_test2.arpa.gz ../nist_lm
mv lm_dev1.arpa.gz ../nist_lm
mv lm_dev2.arpa.gz ../nist_lm


cd "../../../"
