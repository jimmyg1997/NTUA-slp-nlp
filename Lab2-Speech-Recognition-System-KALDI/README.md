# Implementation of a Speech Recognition recognition system (based on phonemes) using the KALDI
## Project Description
The design of the system could further be splitted in the following steps:
* Extraction of Mel-Frequency Cepstral Coefficients (MFCC).
* Language Model Training
* Acoustic Model Training
* Evaluation of the above models seperately and in combination

## Data
* **Database** : USC-timit
* 4 different speakers, with 460 sentences (text, speech) 

## Contents 
* **report.pdf** : The final report of the project which includes:
  * Th design of the speech system, 
  * The procedure followed (bash instructions) since KALDI is difficult in use
  * Evaluation of the different models
  * Answers for the questions set in the **project-description**
  
* utt2spk.py, wav-scp.py, text.py, text2phonem.py: Scripts to create the text files needed by KALDI to train language \& acoustic model

* cmd.sh, path.sh: KALDI scripts that should be placed in the project folder.
