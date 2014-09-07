#include <fst/fstlib.h>
#include "LegacyRnnLMHash.h"
#include "LegacyRnnLMDecodable.h"
#include "LegacyRnnLMReader.h"
#include "RnnLMDecoder.h"
#include <omp.h>
using namespace fst;

template<class H>
void LoadCorpus (std::string& filename,
                 std::vector<std::vector<int> >* corpus, const H& h){
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
      while (ss >> word)
        words.push_back (h.GetWordId (word));
      
      words.push_back (0);
      corpus->push_back (words);
    }
    ifp.close ();
  }
}

void LoadTestSet (const std::string& filename,
		  std::vector<std::vector<std::string> >* corpus,
		  bool rev = false) {
  std::ifstream ifp (filename.c_str ());
  std::string line;

  if (ifp.is_open ()) {
    while (ifp.good ()) {
      getline (ifp, line);
      if (line.empty ())
        continue;
      
      std::string word;
      std::vector<string> words;
      std::stringstream ss (line);
      while (ss >> word)
        words.push_back (word);
      
      if (rev == true)
	reverse (words.begin (), words.end ());

      words.push_back ("</s>");
      corpus->push_back (words);
    }
    ifp.close ();
  }
}

template<class H>
VectorFst<StdArc> WordToFst (const vector<string>& word, H& h) {
  VectorFst<StdArc> fst;
  fst.AddState ();
  fst.SetStart (0);
  for (int i = 0; i < word.size (); i++) {
    int hash = h.HashInput (word.begin () + i,
			    word.begin () + i + 1);
    fst.AddState ();
    fst.AddArc (i, StdArc (hash, hash, StdArc::Weight::One(), i + 1));
  }

  for (int i = 0; i < word.size (); i++) {
    for (int j = 2; j <= 3; j++) {
      if (i + j <= word.size ()) {
	int hash = h.HashInput (word.begin () + i, word.begin () + i + j);
	if (h.imap.find (hash) != h.imap.end ()) 
	  fst.AddArc (i, StdArc (hash, hash, StdArc::Weight::One (), i + j));
      }
    }
  }
  fst.SetFinal (word.size (), StdArc::Weight::One ());

  return fst;
}

typedef LegacyRnnLMDecodable<Token, LegacyRnnLMHash> Decodable;

DEFINE_string (rnnlm,  "", "The input RnnLM model.");
DEFINE_string (test,   "", "The input word list to evaluate.");
DEFINE_int32  (order,   8, "Maximum order for ngram model.");
DEFINE_int32  (nbest,   1, "Maximum number of hypotheses to return.");
DEFINE_int32  (threads, 1, "Number of parallel threads (OpenMP).");
DEFINE_int32  (kmax,   20, "State-local maximum queue size.");
DEFINE_int32  (beam,   20, "The state-local beam width.");
DEFINE_bool   (reverse, false, "Reverse the input word before decoding.");

int main (int argc, char* argv []) {
  string usage = "phonetisaurus-g2prnn --rnnlm test.rnnlm --test test.words --nbest=5\n\n Usage: ";
  set_new_handler (FailedNewHandler);
  SetFlags (usage.c_str (), &argc, &argv, false);
  if (FLAGS_rnnlm.compare ("") == 0 || FLAGS_test.compare ("") == 0) {
    cout << "--rnnlm and --test are required!" << endl;
    exit (1);
  }
    
  omp_set_num_threads (FLAGS_threads);

  vector<vector<string> > corpus;

  LoadTestSet (FLAGS_test, &corpus, FLAGS_reverse);

  typedef unordered_map<int, vector<vector<Chunk> > > RMAP;
  RMAP rmap;
  int csize = corpus.size ();

  LegacyRnnLMReader<Decodable, LegacyRnnLMHash> reader (FLAGS_rnnlm);
  LegacyRnnLMHash h = reader.CopyVocabHash ();
  Decodable s = reader.CopyLegacyRnnLM (h);

  #pragma omp parallel for
  for (int x = 0; x < FLAGS_threads; x++) {
    RnnLMDecoder<Decodable> decoder (s);

    int start = x * (csize / FLAGS_threads);
    int end   = (x == FLAGS_threads - 1) ? csize : start + (csize / FLAGS_threads);
    for (int i = start; i < end; i++) {
      VectorFst<StdArc> fst = WordToFst<LegacyRnnLMHash> (corpus [i], h);
      vector<vector<Chunk> > results = decoder.Decode (fst, FLAGS_beam, FLAGS_kmax, 
						       FLAGS_nbest);
      rmap [i] = results;
    }
  }

  for (int i = 0; i < csize; i++) {
    const vector<vector<Chunk> >& results = rmap [i];
    
    for (int k = 0; k < results.size (); k++) {
      const vector<Chunk>& result = results [k];
      if (FLAGS_reverse == true) {
	for (vector<Chunk>::const_reverse_iterator rit = result.rbegin () + 1;
	     rit != result.rend (); ++rit) 
	  cout << h.vocab_[rit->w].word << " ";
	cout << h.vocab_[result [result.size () - 1].w].word << " ";
      } else {
	for (vector<Chunk>::const_iterator it = result.begin ();
	     it !=result.end (); ++it)
	  cout << h.vocab_[it->w].word << " ";
      }
      cout << result [result.size () - 1].t << "\n";
    }
  }

  return 1;
}
