#include <fst/fstlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include "LegacyRnnLMHash.h"
#include "BRnnLM.h"
using namespace fst;

void LoadVocab (std::string& filename, LegacyRnnLMHash* h){
  std::ifstream ifp (filename.c_str ());
  std::string line;

  if (ifp.is_open ()) {
    while (ifp.good ()) {
      getline (ifp, line);
      if (line.empty ())
	continue;

      std::string word;
      std::stringstream ss (line);
      while (ss >> word) {
	int id = h->FindWord (word);
	if (id == -1)
	  id = h->AddWordToVocab (word);
	else
	  h->vocab_[id].cn++;
      }
      h->vocab_[0].cn++;
    }
    ifp.close ();
  }
}

void LoadCorpus (std::string& filename, 
		 std::vector<std::vector<int> >* corpus, LegacyRnnLMHash* h){
  std::ifstream ifp (filename.c_str ());
  std::string line;

  if (ifp.is_open ()) {
    while (ifp.good ()) {
      getline (ifp, line);
      if (line.empty ())
	continue;

      std::string word;
      std::vector<int> words;
      std::stringstream ss (line);
      while (ss >> word) {
	int id = h->FindWord (word);
	words.push_back (id);
      }
      words.push_back (0);
      corpus->push_back (words);
    }
    ifp.close ();
  }
}

void LearnOneIter (BRnnLM& r, vector<vector<int> >& corpus) {
  //////////////////////////////////
  //Learn for one iteration
  if (r.bptt > 0) 
    for (int i = 0; i < r.bptt + r.bptt_block; i++) 
      r.bptt_history [i] = 0;
  for (int i = 0; i < MAX_NGRAM_ORDER; i++)
    r.history [i] = 0;

  for (int c = 0; c < corpus.size (); c++) {
    int prev = 0;
    for (int i = 0; i < corpus[c].size (); i++) {
      //std::cout << "Counter: " << r.counter_ << std::endl;
      //print bptt history
      /*
      for (int i = MAX_NGRAM_ORDER - 1; i >= 0; i--) 
	std::cout << r.history [i] << " " << r.v_.vocab_[r.history [i]].word << ", ";
      std::cout << std::endl;
      */
      ///////////////////////

      r.ComputeNet (prev, corpus[c][i]);
      // Shift bptt memory for next step
      if (r.bptt > 0) {
	for (int j = r.bptt + r.bptt_block - 1; j > 0; j--) 
	  r.bptt_history[j] = r.bptt_history[j-1];
	r.bptt_history[0] = prev;
	for (int j = r.bptt + r.bptt_block - 1; j > 0; j--) {
	  for (int k = 0; k < r.hsize_; k++) {
	    r.bptt_hidden [j * r.hsize_ + k].ac = 
	      r.bptt_hidden [(j - 1) * r.hsize_ + k].ac;
	    r.bptt_hidden [j * r.hsize_ + k].er = 
	      r.bptt_hidden [(j - 1) * r.hsize_ + k].er;
	  }
	}
      }
      r.LearnNet (prev, corpus[c][i]);
      r.CopyHiddenLayerToInput ();
      //Deactivate the previous word in the input layer. 
      //What purpose does the input layer actually serve?  
      //There is never more than 1 Neuron activated, and the 
      // value is *always* 1 if it is.
      if (prev != -1)
	r.neu0 [prev].ac = 0;
      prev = corpus[c][i];
      for (int i = MAX_NGRAM_ORDER - 1; i > 0; i--) 
	r.history [i] = r.history [i - 1];
      r.history [0] = prev;
    }
    r.NetReset ();
  }
}

void Validate (BRnnLM& r, vector<vector<int> >& valid) {
  //////////////////////////////////
  //Validate for one iteration
  for (int c = 0; c < valid.size (); c++) {
    double prob = r.EvaluateSentence (valid [c]);
    //cout << "S[" << c << "]: " << prob << endl;
  }
}


DEFINE_string (corpus,     "", "The input training corpus.");
DEFINE_string (valid,      "", "The input validation set.");
DEFINE_string (eos,    "</s>", "The end-of-sentence marker.");
DEFINE_string (model,      "", "The output model to write to.");
DEFINE_int32  (max_iter,   10, "The maximum number of training epochs / iterations.");
DEFINE_double (alpha,     0.1, "Basic learning rate.");
DEFINE_double (beta,    1e-07, "Regularisation constant.");
DEFINE_int32  (hsize,      10, "Number of units in the hidden layer.");
DEFINE_int32  (classes,     2, "Number of classes.");
DEFINE_int32  (ndirect,     2, "Number of max-ent direct connections (x1M).");
DEFINE_int32  (order,       2, "Maximum order of max-ent n-gram order.");
DEFINE_int32  (bptt,        2, "Maximum number of BPTT steps. Careful!");
DEFINE_int32  (bptt_block,  2, "BPTT-block for back-propagation. Careful!");
DEFINE_int32  (seed,        1, "Random seed.");

int main (int argc, char* argv []) {
  string usage = "phonetisaurus-train-rnn --corpus=g2p.dic --valid=g2p.valid --model=g2p.rnn\n\n Usage: ";
  set_new_handler (FailedNewHandler);
  SetFlags (usage.c_str (), &argc, &argv, false);

  LegacyRnnLMHash h (FLAGS_classes);

  std::vector<std::vector<int> > corpus;
  std::vector<std::vector<int> > valid;
  h.AddWordToVocab (FLAGS_eos, 0);

  LoadVocab (FLAGS_corpus, &h);
  
  h.SortVocab ();
  h.SetClasses ();
  
  LoadCorpus (FLAGS_corpus, &corpus, &h);
  LoadCorpus (FLAGS_valid, &valid, &h);
  
  BRnnLM r (h, FLAGS_hsize, FLAGS_ndirect * 1000000, 
	    FLAGS_bptt, FLAGS_bptt_block, FLAGS_order,
	    FLAGS_alpha, FLAGS_beta, FLAGS_seed);

  for (int i = 0; i < FLAGS_max_iter; i++) {
    LearnOneIter (r, corpus);
    std::cout << "ITER: " << i << " Max: " << FLAGS_max_iter - 1 << endl;
    if (i < FLAGS_max_iter - 1) {
      std::cout << "Start validation" << std::endl;
      Validate (r, valid);
      std::cout << "End validation" << std::endl;
      std::cout << "ALPHA: 0.1000000000" << std::endl;
      r.ClearActivations ();
      r.NetReset ();
    }
  }
  r.WriteRnnLMModel (FLAGS_model);

  return 1;
}
