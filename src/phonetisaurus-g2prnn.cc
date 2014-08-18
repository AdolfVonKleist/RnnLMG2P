#include <fst/fstlib.h>
#include "LegacyRnnLMHash.h"
#include "LegacyRnnLMDecodable.h"
#include "LegacyRnnLMReader.h"
#include "RnnLMDecoder.h"
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
		  std::vector<std::vector<std::string> >* corpus) {
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
      
      words.push_back ("</s>");
      corpus->push_back (words);
    }
    ifp.close ();
  }
}

template<class H>
VectorFst<StdArc> WordToFst (vector<string>& word, H& h) {
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

DEFINE_string (rnnlm, "", "The input RnnLM model.");
DEFINE_string (test,  "", "The input word list to evaluate.");
DEFINE_int32  (beam,   7, "The local beam width.");
DEFINE_int32  (nbest,  1, "Maximum number of hypotheses to return.");

int main (int argc, char* argv []) {
  string usage = "phonetisaurus-g2prnn --rnnlm test.rnnlm --test test.words --nbest=5\n\n Usage: ";
  set_new_handler (FailedNewHandler);
  SetFlags (usage.c_str (), &argc, &argv, false);

  LegacyRnnLMReader<Decodable, LegacyRnnLMHash> reader (FLAGS_rnnlm);
  LegacyRnnLMHash h = reader.CopyVocabHash ();
  Decodable s = reader.CopyLegacyRnnLM (h);
  RnnLMDecoder<Decodable> decoder (s);
  vector<vector<string> > corpus;
  
  LoadTestSet (FLAGS_test, &corpus);

  for (int i = 0; i < corpus.size (); i++) {
    VectorFst<StdArc> fst = WordToFst<LegacyRnnLMHash> (corpus [i], h);
    vector<vector<Chunk> > results = decoder.Decode (fst, FLAGS_beam, FLAGS_nbest);
    for (int k = 0; k < results.size (); k++) {
      const vector<Chunk>& result = results [k];
      for (int j = 0; j < result.size (); j++) {
	cout << h.vocab_[result [j].w].word << " ";
      }
      cout << result [result.size () - 1].t << endl;
    }
  }
  
  return 1;
}
