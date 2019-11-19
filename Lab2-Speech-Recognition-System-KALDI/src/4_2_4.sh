. ./path.sh # Needed for KALDI_ROOT
. ./cmd.sh

export IRSTLM=$KALDI_ROOT/tools/irstlm/
export PATH=${PATH}:$IRSTLM/bin


./utils/prepare_lang.sh  data/local/dict "<oov>" data/local/lm_tmp data/lang