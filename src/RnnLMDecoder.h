#ifndef RNNLM_DECODER_H__
#define RNNLM_DECODER_H__

#include <fst/fstlib.h>
#include "LegacyRnnLMDecodable.h"
#include "LegacyRnnLMHash.h"
#include <vector>
#include <unordered_set>

using fst::VectorFst;
using fst::ArcIterator;
using fst::StdArc;
using fst::Heap;
using std::vector;
using std::unordered_set;

class Token {
 public:
  Token (int hsize, int max_order) 
    : word (0), weight (0.0), total (0.0), 
      g (0.0), prev (NULL), state (0), key (-1) {
    hlayer.resize (hsize, 1.0);
    history.resize (max_order, 0);

    HashHistory ();
  }

  Token (Token* tok, int w, int s) 
    : word (w), weight (0.0), total (0.0), 
      g (0.0), prev (tok), state (s), key (-1) {
    //Copy an existing token and update the 
    // various layers as needed
    hlayer.resize (tok->hlayer.size(), 0.0);
    history.resize (tok->history.size (), 0);

    //Would it be more efficient to perform the hash 
    // by iterating back throug the parent tokens?
    for (int i = tok->history.size () - 1; i > 0; i--)
      history [i] = tok->history [i - 1];
    history [0] = tok->word;

    HashHistory ();
  }

  void HashHistory () {
    hhash = state * 7853;
    for (int i = 0; i < 7; i++)
      hhash = hhash * 7877 + history [i];
  }

  int state;
  int word;
  mutable int key;
  mutable double weight;
  mutable double g;
  mutable double total;
  mutable Token* prev;
  mutable vector<double> hlayer;
  mutable vector<int> history;
  size_t hhash;
};

class TokenCompare {
 public:
  bool operator () (const Token& t1, const Token& t2) const {
    return (t1.state == t2.state &&
	    t1.word == t2.word &&
	    t1.hhash == t2.hhash);
    /*
    return (t1.state == t2.state &&
	    t1.word == t2.word);
    */
  }
};

class TokenHash {
 public:
  size_t operator () (const Token& t) const {
    return t.state * kPrime0 + t.word * kPrime1 + t.hhash * kPrime2;
    //return t.state * kPrime0 + t.word * kPrime1;
  }
 private:
  static const size_t kPrime0;
  static const size_t kPrime1;
  static const size_t kPrime2;
};
const size_t TokenHash::kPrime0 = 7853;
const size_t TokenHash::kPrime1 = 7867;
const size_t TokenHash::kPrime2 = 7873;


class TokenPointerCompare {
 public:
  bool operator () (const Token* t1, const Token* t2) const {
    return (t1->g < t2->g);
  }
};

struct Chunk {
 public:
  Chunk (int word, double cost, double total) 
    : w (word), c (cost), t (total) { }
  int w;
  double c;
  double t;
};

template <class D>
class RnnLMDecoder {
 public:
  typedef D Decodable;
  typedef vector<vector<Chunk> > Results;
  typedef Heap<Token*, TokenPointerCompare, false> Queue;
  typedef unordered_set<Token, TokenHash, TokenCompare> TokenSet;

  RnnLMDecoder (Decodable& decodable) 
    : d (decodable) { }

  double Heuristic (int nstate, int nstates, double hcost) {
    int factor = nstates - nstate - 1;
    if (factor > 0) 
      return factor * hcost;
    return 0.0;
  }

  Results Decode (VectorFst<StdArc>& fst, int beam, int kMax, int nbest, double hcost) {
    Initialize ();
    int n = 0;
    double best;
    Queue sQueue;
    TokenSet sPool;

    while (!queue.Empty () && n < nbest) {
      Token* top = queue.Pop ();
      if (fst.Final (top->state) != StdArc::Weight::Zero ()) {
	Token* a = (Token*)&(*top);
	vector<Chunk> result;
	while (a->prev != NULL) {
	  result.push_back (Chunk (a->word, a->weight, a->total));
	  a = (Token*)a->prev;
	}
	reverse (result.begin (), result.end ());
	results.push_back (result);
	n++;
	continue;
      }

      for (ArcIterator<VectorFst<StdArc> > aiter (fst, top->state);
	   !aiter.Done (); aiter.Next ()) {
	const StdArc& arc = aiter.Value ();
	const vector<int>& map = d.h.imap [arc.ilabel];
	sQueue.Clear ();

	best = 999;
	for (int i = 0; i < map.size (); i++) {
	  Token nexttoken ((Token*)&(*top), map [i], arc.nextstate);
	  nexttoken.weight = -log (d.ComputeNet ((*top), &nexttoken));
	  nexttoken.total += top->total + nexttoken.weight;
	  nexttoken.g = nexttoken.total + Heuristic (arc.nextstate, fst.NumStates (), hcost);
	  if (nexttoken.weight < best || abs (nexttoken.weight - best) < beam) {
	    TokenSet::iterator nextiterator = pool.find (nexttoken);

	    if (nextiterator == pool.end ()) {
	      pool.insert (nexttoken);
	      Token* nextpointer = (Token*)&(*pool.find (nexttoken));
	      //nextpointer->key = queue.Insert (nextpointer);
	      sQueue.Insert (nextpointer);
	    } else {
	      //cout << "Found: " << arc.ilabel << " : " << nexttoken.word << endl;
	      if (nexttoken.total < nextiterator->total) {
		nextiterator->weight  = nexttoken.weight;
		nextiterator->total   = nexttoken.total;
		nextiterator->prev    = nexttoken.prev;
		nextiterator->history = nexttoken.history;
		nextiterator->g       = nexttoken.g;
		nextiterator->hlayer  = nexttoken.hlayer;
		queue.Update (nextiterator->key, (Token*)&(*nextiterator));
	      }
	    }

	    if (nexttoken.weight < best)
	      best = nexttoken.weight;
	  }
	}

	int k = 0;
	while (!sQueue.Empty () && k < kMax) {
	  Token* tpointer = sQueue.Pop ();
	  tpointer->key = queue.Insert (tpointer);
	  k++;
	}
      }
    }

    return results;
  }

  Results  results;


 private:
  void Initialize () {
    queue.Clear ();
    pool.clear ();
    results.clear ();

    Token start (d.hsize, d.max_order);
    pool.insert (start);
    TokenSet::iterator prev = pool.find (start);
    prev->key = queue.Insert ((Token*)&(*prev));
    return;
  }

  Decodable& d;
  Queue    queue;
  TokenSet pool;
};
#endif // RNNLM_DECODER_H__
