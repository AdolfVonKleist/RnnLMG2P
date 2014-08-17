#ifndef BRNNHASH_H__
#define BRNNHASH_H__

#include <fst/fstlib.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <math.h>
#include <sstream>


typedef double real;

struct VocabWord {
 public:
  VocabWord () {}
  VocabWord (std::string word_) : cn (1), word (word_) {}
  VocabWord (std::string word_, int cn_) : cn (cn_), word (word_) {}
  int    cn;   //Unigram count
  std::string word;
  real   prob;
  int    class_index;
};

struct ClassIndex {
 public:
  ClassIndex () : begin(0), end(0) {}
  int begin;
  int end;
};

class BRnnHash {
 public:
  BRnnHash (int class_size) : class_size_ (class_size) {
    vocab_hash_.resize (100000000);
  }

  static const std::vector<unsigned int> primes_;
  static const char* g_delim;
  static const char* gp_delim;

  void Split (const std::string& s, char delim, std::vector<std::string>& elems) {
    std::stringstream ss (s);
    std::string item;
    while (getline (ss, item, delim))
      elems.push_back (item);
  }

  template <typename I>
  int HashInput (I start, I end) {
    size_t hash = 0;
    for (I it = start; it != end; ++it)
      hash = hash * 237 + isyms.Find (*it);
    
    return hash;
  }

  void MapToken (string& token) {
    std::vector<std::string> gp;
    std::vector<std::string> graphs;

    Split (token, *gp_delim, gp);
    Split (gp [0], *g_delim, graphs);

    size_t hash = 0;
    for (int i = 0; i < graphs.size (); i++)
      hash = hash * 237 + isyms.AddSymbol (graphs [i]);

    if (imap.find (hash) == imap.end ())
      imap [hash] = std::vector<int> {FindWord (token)};
    else
      imap [hash].push_back (FindWord (token));
  }

  int HashWord (std::string& word) const {
    size_t hash = 0;
    for (size_t i = 0; i < word.size (); i++)
      hash = hash * 237 + word[i];
    hash = hash % vocab_hash_.size ();
    return hash;
  }

  int FindWord (std::string& word) {
    size_t hash = HashWord (word);
    
    if (vocab_hash_[hash] == -1)
      return -1;

    if (word.compare (vocab_[vocab_hash_[hash]].word) == 0)
      return vocab_hash_[hash];

    for (size_t i = 0; i < vocab_.size (); i++) {
      if (word.compare (vocab_[i].word) == 0) {
	vocab_hash_[hash] = i;
	return i;
      }
    }
    return -1;
  }

  int GetWordId (std::string& word) const {
    size_t hash = HashWord (word);
    if (vocab_hash_[hash] == -1)
      return -1;
    return vocab_hash_[hash];
  }

  int AddWordToVocab (std::string& word, int cn = 1) {
    vocab_.push_back (VocabWord (word, cn));
    size_t hash = HashWord (word);
    vocab_hash_[hash] = vocab_.size () - 1;
    return vocab_.size () - 1;
  }

  void SortVocab () {
    // Just sorts based on Class
    for (int i = 1; i < vocab_.size (); i++) {
      int max = i;
      for (int j = i + 1; j < vocab_.size (); j++)
	if (vocab_[max].cn < vocab_[j].cn)
	  max = j;
      VocabWord swap = vocab_[max];
      vocab_[max] = vocab_[i];
      vocab_[i]   = swap;
    }
  }
  
  void SetClasses () {
    double df = 0;
    double dd = 0;
    int     a = 0;
    int     b = 0;

    for (int i = 0; i < vocab_.size (); i++)
      b += vocab_[i].cn;
    for (int i = 0; i < vocab_.size (); i++)
      dd += sqrt (vocab_[i].cn / (double)b);
    for (int i = 0; i < vocab_.size (); i++) {
      df += sqrt (vocab_[i].cn / (double)b) / dd;
      if (df > 1)
	df = 1;
      if (df > (a + 1) / (double)class_size_) {
	vocab_[i].class_index = a;
	if (a < class_size_ - 1)
	  a++;
      } else {
	vocab_[i].class_index = a;
      }
    }

    class_sizes_.resize (class_size_);
    int c = 0;
    for (int i = 0; i < vocab_.size (); i++) {
      if (i == 0) {
	class_sizes_[c].begin = i;
      }

      if (i + 1 == vocab_.size ()) {
	class_sizes_[c].end = i;
      } else if (vocab_[i].class_index < vocab_[i + 1].class_index) {
	class_sizes_[c].end = i;
	c++;
	class_sizes_[c].begin = i + 1;
      }
    }
  }

  std::vector<size_t> vocab_hash_;
  std::vector<VocabWord> vocab_;
  std::vector<ClassIndex> class_sizes_;
  std::unordered_map<int, std::vector<int> > imap;
  fst::SymbolTable isyms;
  int class_size_;
};

const std::vector<unsigned int> BRnnHash::primes_ = {
  108641969, 116049371, 125925907, 133333309,
  145678979, 175308587, 197530793, 234567803,
  251851741, 264197411, 330864029, 399999781,
  407407183, 459258997, 479012069, 545678687,
  560493491, 607407037, 629629243, 656789717,
  716048933, 718518067, 725925469, 733332871,
  753085943, 755555077, 782715551, 790122953,
  812345159, 814814293, 893826581, 923456189,
  940740127, 953085797, 985184539, 990122807
};

const char* BRnnHash::g_delim  = "|";
const char* BRnnHash::gp_delim = "}";

#endif // BRNNHASH_H__