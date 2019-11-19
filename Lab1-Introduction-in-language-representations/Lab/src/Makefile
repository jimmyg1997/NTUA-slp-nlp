orth_I:
	fstcompile --isymbols=chars.syms --osymbols=chars.syms orth_I.txt orth_I.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms orth_I.fst orth_I.dot
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait orth_I.fst | dot -Tjpg >orth_I.jpg

orth_E: 
	fstcompile --isymbols=chars.syms --osymbols=chars.syms orth_E.txt orth_E.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms orth_E.fst orth_E.dot
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait orth_E.fst | dot -Tjpg >orth_E.jpg


transducer:
	fstconcat orth_I.fst orth_E.fst transducer.fst
	fstconcat transducer.fst orth_I.fst transducer.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms transducer.fst transducer.dot
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait transducer.fst | dot -Tjpg >transducer.jpg

transducershortest:
	fstshortestpath transducer.fst transducershortest.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms transducershortest.fst transducershortest.dot
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait transducershortest.fst | dot -Tjpg >transducershortest.jpg

orth_acceptor:
	fstcompile --isymbols=chars.syms --osymbols=chars.syms orth_acceptor.txt orth_acceptor.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms orth_acceptor.fst orth_acceptor.dot

orth_acceptor_processed:
	fstrmepsilon orth_acceptor.fst orth_acceptor_processed.fst
	fstdeterminize orth_acceptor_processed.fst orth_acceptor_processed.fst
	fstminimize orth_acceptor_processed.fst orth_acceptor_processed.fst

orthograph:
	fstarcsort --sort_type=olabel transducer.fst transducer.fst
	fstarcsort --sort_type=ilabel orth_acceptor_processed.fst orth_acceptor_processed.fst
	fstcompose transducer.fst orth_acceptor_processed.fst orthograph.fst
	#fstcompose transducer.fst orth_acceptor_processed.fst | fstprint --isymbols=chars.syms --osymbols=chars.syms
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms orthograph.fst orthograph.dot 


check_cit:
	fstcompile --isymbols=chars.syms --acceptor=true cit.txt cit.fst
	fstcompose cit.fst orthograph.fst possible_words.fst
	fstshortestpath possible_words.fst possible_words.fst
	fstrmepsilon possible_words.fst possible_words.fst
	fstprint --isymbols=chars.syms --osymbols=chars.syms possible_words.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait possible_words.fst | dot -Tjpg >possible_words.jpg



#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------STEP 11 b -------------#

orth_I_words:
	fstcompile --isymbols=chars.syms --osymbols=chars.syms orth_I_words.txt orth_I_words.fst

orth_E_words: 
	fstcompile --isymbols=chars.syms --osymbols=chars.syms orth_E_words.txt orth_E_words.fst

transducer_words:
	fstconcat orth_I_words.fst orth_E_words.fst transducer_words.fst
	fstconcat transducer_words.fst orth_I_words.fst transducer_words.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait transducer_words.fst | dot -Tjpg >transducer_words.jpg

transducershortest_words:
	fstshortestpath transducer_words.fst transducershortest_words.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait transducershortest_words.fst | dot -Tjpg >transducershortest_words.jpg


#---------------------STEP 11 c -------------#
orth_I_chars:
	fstcompile --isymbols=chars.syms --osymbols=chars.syms orth_I_chars.txt orth_I_chars.fst

orth_E_chars: 
	fstcompile --isymbols=chars.syms --osymbols=chars.syms orth_E_chars.txt orth_E_chars.fst

transducer_chars:
	fstconcat orth_I_chars.fst orth_E_chars.fst transducer_chars.fst
	fstconcat transducer_chars.fst orth_I_chars.fst transducer_chars.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait transducer_chars.fst | dot -Tjpg >transducer_chars.jpg


transducershortest_chars:
	fstshortestpath transducer_chars.fst transducershortest_chars.fst


#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------STEP 12 a,b -------------#
orth_acceptor_words:
	fstcompile --isymbols=chars.syms --osymbols=chars.syms orth_acceptor_words.txt orth_acceptor_words.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait orth_acceptor_words.fst | dot -Tjpg >orth_acceptor_words.jpg

orth_acceptor_processed_words:
	fstrmepsilon orth_acceptor_words.fst orth_acceptor_processed_words.fst
	fstdeterminize orth_acceptor_processed_words.fst orth_acceptor_processed_words.fst
	fstminimize orth_acceptor_processed_words.fst orth_acceptor_processed_words.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait orth_acceptor_processed_words.fst | dot -Tjpg >orth_acceptor_processed_words.jpg

#---------------------STEP 12 c -------------#


orth_acceptor_chars:
	fstcompile --isymbols=chars.syms --osymbols=chars.syms orth_acceptor_chars.txt orth_acceptor_chars.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait orth_acceptor_chars.fst | dot -Tjpg >orth_acceptor_chars.jpg

orth_acceptor_processed_chars:
	fstrmepsilon orth_acceptor_chars.fst orth_acceptor_processed_chars.fst
	fstdeterminize orth_acceptor_processed_chars.fst orth_acceptor_processed_chars.fst
	fstminimize orth_acceptor_processed_chars.fst orth_acceptor_processed_chars.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms --fontsize=16 -portrait orth_acceptor_processed_chars.fst | dot -Tjpg >orth_acceptor_processed_chars.jpg



#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------STEP 13 a -------------#

orthograph_words:
	fstarcsort --sort_type=olabel transducer_words.fst transducer_words.fst
	fstarcsort --sort_type=ilabel orth_acceptor_processed_words.fst orth_acceptor_processed_words.fst
	fstcompose transducer_words.fst orth_acceptor_processed_words.fst orthograph_words.fst
	#fstcompose transducer.fst orth_acceptor_processed.fst | fstprint --isymbols=chars.syms --osymbols=chars.syms
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms orthograph.fst orthograph.dot 

#---------------------STEP 13 b -------------#
orthograph_chars:
	fstarcsort --sort_type=olabel transducer_words.fst transducer_words.fst
	fstarcsort --sort_type=ilabel orth_acceptor_processed_chars.fst orth_acceptor_processed_chars.fst
	fstcompose transducer_words.fst orth_acceptor_processed_chars.fst orthograph_chars.fst
	#fstcompose transducer.fst orth_acceptor_processed.fst | fstprint --isymbols=chars.syms --osymbols=chars.syms
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms orthograph.fst orthograph.dot 

#---------------------STEP 13 c -------------#
check_cit_words:
	fstcompile --isymbols=chars.syms --acceptor=true cit.txt cit.fst
	fstcompose cit.fst orthograph_words.fst possible_words_words.fst
	fstshortestpath possible_words_words.fst possible_words_words.fst
	fstrmepsilon possible_words_words.fst possible_words_words.fst
	#fstarcsort --sort_type=olabel possible_words_words.fst possible_words_words.fst
	fsttopsort | fstprint -osymbols=chars.syms possible_words_words.fst | cut -f4 | grep -v "<epsilon>" 

check_cit_chars:
	fstcompose cit.fst orthograph_chars.fst possible_words_chars.fst
	fstshortestpath possible_words_chars.fst possible_words_chars.fst
	fstrmepsilon possible_words_chars.fst possible_words_chars.fst
	#fstarcsort --sort_type=olabel possible_words_chars.fst possible_words_chars.fst
	fsttopsort | fstprint -osymbols=chars.syms possible_words_chars.fst | cut -f4 | grep -v "<epsilon>" 

#-------------------------------------------------------------------------------------------------------------------------------#
#---------------------STEP 13 b -------------#

unique_word:
	fstcompile --isymbols=chars.syms --osymbols=chars.syms --acceptor word_acceptor.txt word_acceptor.fst
	fstarcsort --sort_type=olabel word_acceptor.fst word_acceptor.fst

	fstarcsort --sort_type=ilabel orthograph_words.fst orthograph_words.fst
	fstcompose word_acceptor.fst  orthograph_words.fst result_words.fst
	fstrmepsilon result_words.fst result_words.fst
	fstshortestpath result_words.fst resultshorterst_words.fst
	fstrmepsilon result_words.fst resultshorterst_words.fst
	fstarcsort result_words.fst resultshorterst_words.fst
	fstprint resultshorterst_words.fst > Acceptor_Shortest_words.txt

	fstarcsort --sort_type=ilabel orthograph_chars.fst orthograph_chars.fst
	fstcompose word_acceptor.fst  orthograph_chars.fst result_chars.fst
	fstrmepsilon result_chars.fst result_chars.fst
	fstshortestpath result_chars.fst resultshorterst_chars.fst
	fstrmepsilon result_chars.fst resultshorterst_chars.fst
	fstarcsort result_chars.fst resultshorterst_chars.fst
	fstprint resultshorterst_chars.fst > Acceptor_Shortest_chars.txt
	



