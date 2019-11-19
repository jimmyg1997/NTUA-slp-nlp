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

unique_word:
	fstcompile --isymbols=chars.syms --osymbols=chars.syms --acceptor word_acceptor.txt word_acceptor.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms word_acceptor.fst word_acceptor.dot
	fstarcsort --sort_type=olabel word_acceptor.fst word_acceptor.fst
	fstarcsort --sort_type=ilabel orthograph.fst orthograph.fst
	fstcompose word_acceptor.fst  orthograph.fst result.fst
	fstrmepsilon result.fst result.fst
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms result.fst result.dot
	fstshortestpath result.fst resultshorterst.fst
	fstrmepsilon result.fst resultshorterst.fst
	fstarcsort result.fst resultshorterst.fst
	fstprint resultshorterst.fst > Acceptor_Shortest.txt
	#fstdraw --isymbols=chars.syms --osymbols=chars.syms resultshorterst.fst resultshorterst.dot



