#!/bin/bash

# Copyright 2013  (Author: Daniel Povey)
# Apache 2.0

# This script takes data prepared in a corpus-dependent way
# in data/local/, and converts it into the "canonical" form,
# in various subdirectories of data/, e.g. data/lang, data/train, etc.

. ./path.sh || exit 1;

echo "Preparing train, dev and test data"
srcdir=data/local/dict
lmdir=data/local/nist_lm
tmpdir=data/local/lm_tmp
lexicon=data/local/dict/lexicon.txt
mkdir -p $tmpdir



# Next, for each type of language model, create the corresponding FST
# and the corresponding lang_test_* directory.

echo "Preparing language models for test"

for x in train dev test; do
  test=data/lang_test
  mkdir -p $test
  cp -r data/lang/* $test


  gunzip -c $lmdir/lm_${x}1.arpa.gz \
   | arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$test/words.txt - data/lang_test/G_${x}1.fst
 # The output is like:
 # 9.14233e-05 -0.259833
 # we do expect the first of these 2 numbers to be close to zero (the second is
 # nonzero because the backoff weights make the states sum to >1).
 # Because of the <s> fiasco for these particular LMs, the first number is not
 # as close to zero as it could be.

done



for x in train dev test; do
  test=data/lang_test  
 
  gunzip -c $lmdir/lm_${x}2.arpa.gz \
   | arpa2fst --disambig-symbol=#0 \
             --read-symbol-table=$test/words.txt - data/lang_test/G_${x}2.fst
 # The output is like:
 # 9.14233e-05 -0.259833
 # we do expect the first of these 2 numbers to be close to zero (the second is
 # nonzero because the backoff weights make the states sum to >1).
 # Because of the <s> fiasco for these particular LMs, the first number is not
 # as close to zero as it could be.

done

#utils/validate_lang.pl data/lang_test || exit 1

echo "Succeeded in formatting data."

