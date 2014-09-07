#ifndef BRNNLM_H__
#define BRNNLM_H__

#include <vector>
#include <iostream>
#include <iomanip>
#include <fstream>
#include "LegacyRnnLMHash.h"

static union{
  double d;
  struct{
    int j,i;
  } n;
} d2i;
#define EXP_A (1048576/M_LN2)
#define EXP_C 60801
#define FAST_EXP(y) (d2i.n.i = EXP_A*(y)+(1072693248-EXP_C),d2i.d)

const int MAX_NGRAM_ORDER=20;

struct Neuron {
  Neuron () : ac(0), er(0) {}

  void Clear (real ac_ = 0.0) {
    ac = ac_;
    er = 0.0;
  }

  real ac; //Activation
  real er; //Error residue
};

struct Synapse {
  real weight; //Link weight
};

class BRnnLM {
 public:
  BRnnLM (const LegacyRnnLMHash& vocab, int hsize, int ndirect, 
	  int bptt_, int bptt_block_, int dorder, double alpha, 
	  double beta, int seed = 1) 
    : v_(vocab), hsize_(hsize), ndirect_(ndirect), bptt(bptt_),
      bptt_block (bptt_block_), direct_order (dorder), 
      alpha_(alpha), beta_(beta) { 
    srand (seed);
    counter_ = 0;
    //Layer sizes
    isize_ = v_.vocab_.size () + hsize_;
    osize_ = v_.vocab_.size () + v_.class_size_;
    //Network layers
    neu0.resize (isize_);
    for (int i = isize_ - hsize_; i < isize_; i++) 
      neu0 [i].ac = 0.1;

    neu1.resize (hsize_);
    neu2.resize (osize_);
    //Connections
    //std::cout << "Syn0:" << std::endl;
    syn0.resize (isize_ * hsize_);
    for (int j = 0; j < hsize_; j++) {
      for (int i = 0; i < isize_; i++) {
	syn0[i + j * isize_].weight = 
	  Random (-0.1, 0.1) + Random (-0.1, 0.1) + Random (-0.1, 0.1);
	/*
	std::cout << "i: " << i << " j: " << j 
		  << " val: " << syn0[i + j * isize_].weight 
		  << std::endl;
	*/
      }
    }

    //std::cout << "Syn1:" << std::endl;
    syn1.resize (hsize_ * osize_);
    for (int j = 0; j < osize_; j++) {
      for (int i = 0; i < hsize_; i++) {
	syn1[i + j * hsize_].weight = 
	  Random (-0.1, 0.1) + Random (-0.1, 0.1) + Random (-0.1, 0.1);
	/*
	std::cout << "i: " << i << " j: " << j
		  << " val: " << syn1[i + j * hsize_].weight
		  << std::endl;
	*/
      }
    }

    //Direct connection synapses should be stored in a hash table.
    // This way of doing things is crazy inefficient from a RAM/storage perspective.
    synd.resize (ndirect_);
    for (int i = 0; i < synd.size (); i++) 
      synd [i].weight = 0.0;
    std::cout << "ndirect: " << synd.size () << std::endl;

    if (bptt > 0) {
      bptt_history.resize (bptt + bptt_block + 10);
      for (int i = 0; i < bptt + bptt_block; i++)
	bptt_history[i] = -1;

      bptt_hidden.resize ((bptt + bptt_block + 1) * hsize_);
      //std::cout << "BPTT hidden: " << bptt_hidden.size () << std::endl;
      bptt_syn0.resize (isize_ * hsize_);
      //std::cout << "BPTT syn: " << bptt_syn0.size () << std::endl;
    }
    for (int i = 0; i < MAX_NGRAM_ORDER; i++)
      history.push_back (0);

    bool verbose = false;
    if (verbose == true) {
      std::cout << "bptt: " << bptt << std::endl;
      std::cout << "bptt_block: " << bptt_block << std::endl;
      std::cout << "vocab: " << v_.vocab_.size () << std::endl;
      std::cout << "input: " << isize_ << std::endl;
      std::cout << "hidden: " << hsize_ << std::endl;
      std::cout << "output: " << osize_ << std::endl;
    }
  }

  real Random (real min, real max) {
    return rand () / (real)RAND_MAX * (max - min) + min;
  }

  double ComputeNet (int previous, int current) {
    //Activate the 'previous' word
    if (previous != -1)
      neu0 [previous].ac = 1;

    //std::cout << "Input layer activations: " << std::endl;
    for (int i = 0; i < hsize_; i++) {
      neu1[i].ac = 0;
      //std::cout << i << ": " << neu1[i].ac << " ";
    }
    //std::cout << std::endl;
    //First compute the activations from the previous hidden layer (zero if we are starting)
    MatrixXVectorActivate (&neu0, &neu1, syn0, 0, hsize_, isize_ - hsize_, isize_);

    //std::cout << "Hidden layer:" << std::endl;
    //Augment the activation with the weights from the previous word
    for (int i = 0; i < hsize_; i++) {
      if (previous != -1) {
	//If neu0 [previous] is always 1, why bother with lookup and multiply??
	// does the compiler optimize this out?
	neu1[i].ac += neu0 [previous].ac * syn0 [previous + i * isize_].weight;
      }
    }
    
    //Floor/Ciel all values
    for (int i = 0; i < hsize_; i++) {
      if (neu1 [i].ac > 50)
	neu1 [i].ac = 50;
      if (neu1 [i].ac < -50)
	neu1 [i].ac = -50;
      real val = -neu1[i].ac;
      //Sigmoid
      neu1[i].ac = 1 / (1 + FAST_EXP (val));
      //std::cout << i << ": " << neu1[i].ac << " ";
    }
    //std::cout << std::endl;
    
    //1 -> 2 class.  Clear all values
    for (int i = v_.vocab_.size (); i < osize_; i++)
      neu2 [i].ac = 0;
    
    //Activate classes
    MatrixXVectorActivate (&neu1, &neu2, syn1, v_.vocab_.size (), osize_, 0, hsize_);
    
    ////////////////////////////////////////////////////
    //Begin direct connection activations for classes
    if (ndirect_ > 0) {
      //Pointers to the synd direct-connection synapses
      unsigned long long hash [MAX_NGRAM_ORDER];
      
      //Clear all previous hash activations
      for (int i = 0; i < direct_order; i++)
	hash [i] = 0;

      //////////////////////////////////////
      //Begin compute feature hash
      for (int i = 0; i < direct_order; i++) {
	//If OOV was in history, do not use this n-gram feature or higher orders
	// WHY?  This should work as long as the <oov> class exists.
	if (i > 0)
	  if (history [i - 1] == -1)
	    break;
	//Otherwise, start computing the hash for this feature
	hash [i] = v_.primes_[0] * v_.primes_[1];

	//Continue computing the hash using words from the history
	for (int j = 1; j <= i; j++) {
	  hash [i] +=
	    v_.primes_ [(i * v_.primes_ [j] + j) % v_.primes_.size ()]
	    * (unsigned long long) (history [j - 1] + 1);
	}
	
	//Make sure that the starting hash index is in the first
	// half of the synd direct-connection synapse vector.
	//The final result is an index into the synd vector
	hash [i] = hash [i] % (ndirect_ / 2);
      }
      //End compute feature hash
      //////////////////////////////////////
      
      //Run through all class-neurons in the output layer
      for (int i = v_.vocab_.size (); i < osize_; i++) {
	//Run through the features in n-gram order
	for (int j = 0; j < direct_order; j++) {
	  if (hash [j]) {
	    //Apply current parameter and move on to the next one.
	    //The class output neuron activation is updated with the 
	    // synapse weight for the hash-selected synapse for the
	    // current ngram-order feature.
	    neu2 [i].ac += synd [hash [j]].weight;
	    //The value of the hash for this order is incremented
	    //WHY? My best guess is that this is just to further 
	    // minimize collisions in the hashing space.  It means
	    // that when we move to the next 'class' index, that 
	    // the value of 'hash [j]' will not be the same one 
	    // that we encountered searching for the previous class
	    // neuron activation.
	    hash [j]++;
	    //In other words, if we do not increment the synapse index
	    // here, then for an output layer with 2 classes, when 
	    // we cycle through the direct_order values, we would get
	    //For class1 := i = 0 && order := j = 1:
	    //    hash [j] = X
	    //For class2 := i = 1 && order := j = 1:
	    //    hash [j] = Y
	    // where X == Y meaning that we would be using the same
	    // synd direct synapse indices for all class-neurons
	    // for a particular direct_order / ME feature.
	  } else {
	    break;
	  }
	}
      }
    }
    //End direct connection activations for classes
    /////////////////////////////////////////////////

    //Softmax on classes
    real sum = 0;
    for (int i = v_.vocab_.size (); i < osize_; i++) {
      if (neu2[i].ac > 50)
	neu2[i].ac = 50;
      if (neu2[i].ac < -50)
	neu2[i].ac = -50;
      real val = FAST_EXP (neu2 [i].ac);
      sum += val;
      neu2[i].ac = val;
    }
    for (int i = v_.vocab_.size (); i < osize_; i++)
      neu2[i].ac /= sum;

    //1 -> 2 word Activations
    if (current != -1) {
      //This is the index range for the WORDS in Class X, in the 
      // final output layer of the network
      int begin = v_.class_sizes_[v_.vocab_[current].class_index].begin;
      int end   = v_.class_sizes_[v_.vocab_[current].class_index].end;
      //Again, reset everything
      for (int i = begin; i <= end; i++)
	neu2 [i].ac = 0;
      //std::cout << "1->2 Word activations" << " Begin: " << begin << " End: " << end << std::endl;
      MatrixXVectorActivate (&neu1, &neu2, syn1, begin, end + 1, 0, hsize_);
      
      ///////////////////////////////////////////////
      //Begin direct connection activations for words
      if (ndirect_ > 0) {
	unsigned long long hash [MAX_NGRAM_ORDER];
	//Reset the hash pointers
	for (int i = 0; i < direct_order; i++)
	  hash [i] = 0;
	///////////////////////////////////////
	//Begin hash computation for words
	//Compute hash pointers to direct order synapses
	for (int i = 0; i < direct_order; i++) {
	  if (i > 0) 
	    if (history [i - 1] == -1)
	      break;
	  //Start with the current word-class index.  Note the difference
	  // to the hashing function used to initialize the class-activations
	  hash [i] = 
	    v_.primes_ [0] * v_.primes_ [1]
	    * (unsigned long long) (v_.vocab_[current].class_index + 1);
	  //Continue computing the hash for all n-gram features less-than-or-equal-to
	  // the current order.  Again the idea is to build an 'n-gram hash' based 
	  // on the preceding word histories.  
	  for (int j = 1; j <= i; j++) {
	    hash [i] += 
	      v_.primes_ [(i * v_.primes_ [j] + j) % v_.primes_.size ()]
	      * (unsigned long long) (history [j - 1] + 1);
	  }
	  //Finalize the hash computation.  Make sure these indices end up
	  // in the second half of the synd vector, which we associate with 
	  // the 'words' as opposed to the 'classes' in the output layer.
	  hash [i] = (hash [i] % (ndirect_ / 2)) + (ndirect_ / 2);
	}
	//End hash computation for words
	//////////////////////////////////////////

	for (int i = begin; i <= end; i++) {
	  for (int j = 0; j < direct_order; j++) {
	    if (hash [j]) {
	      //std::cout << "Hash: " << j << " " << hash [j] << std::endl;
	      neu2 [i].ac += synd [hash [j]].weight;
	      hash [j]++;
	      //What is the idea here?  It is not guaranteed to be in the
	      // second half of synd connections vector after this, it is
	      // only guaranteed not to overflow...
	      hash [j] = hash [j] % ndirect_;
	    } else {
	      break;
	    }
	  }
	}
      }
      //End direct connection activations for words
      //////////////////////////////////////////////

      sum = 0;
      for (int i = begin; i <= end; i++) {
	if (neu2[i].ac > 50)
	  neu2[i].ac = 50;
	if (neu2[i].ac < -50)
	  neu2[i].ac = -50;
	real val = FAST_EXP (neu2[i].ac);
	sum += val;
	neu2[i].ac = val;
      }
      for (int i = begin; i <= end; i++) {
	neu2[i].ac /= sum;
	//std::cout << "Activated: " << i << " : " << neu2[i].ac << std::endl;
      }
    }
    //Output the actual prob now [P(w|c) * P(c)]
    double wGc = neu2 [current].ac;
    double c   = neu2 [v_.vocab_.size () + v_.vocab_[current].class_index].ac;
    double w   = wGc * c;
    /*
    std::cout << "Current: "  << current << ", Prev: " << previous << "\t";
    std::cout << " P (w|C): " << wGc 
	      << " P (C): "   << c 
	      << " P (w) = "  << w
	      << std::endl;
    */
    return w;
  }

  void MatrixXVectorActivate (std::vector<Neuron>* src, std::vector<Neuron>* dest,
			      std::vector<Synapse>& srcmat, int sbegin, int send,
			      int dbegin, int dend) {
    //std::cout << sbegin << ", " << send << ", " << dbegin << ", " << dend << std::endl;
    //std::cout.precision (10);
    //std::cout << "MVals: " << std::endl;
    for (int j = sbegin; j < send; j++) {
      for (int i = dbegin; i < dend; i++) {
	//std::cout << "BEFORE: " << std::fixed << (*dest)[j].ac;
	(*dest)[j].ac += (*src)[i].ac * srcmat[i + j * src->size ()].weight;
	/*
	std::cout << " ACTIVATE: i: " << i << " j: " << j << " val: " << (*dest)[j].ac 
		  << " .. (" << (*src)[i].ac << " * "
		  << srcmat [i + j * src->size ()].weight << ")" << std::endl;
	*/
      }
    }
  }

  void MatrixXVectorError (std::vector<Neuron>* src, std::vector<Neuron>* dest,
			  std::vector<Synapse>& srcmat, int sbegin, int send,
			  int dbegin, int dend) {
    //std::cout << sbegin << ", " << send << ", " << dbegin << ", " << dend << std::endl;
    //std::cout << "MVals: " << std::endl;
    for (int i = dbegin; i < dend; i++) {
      for (int j = sbegin; j < send; j++ ) {
	(*dest)[i].er += (*src)[j].er * srcmat [i + j * dest->size ()].weight;
        /*
	std::cout << "ERROR: i: " << i << " j: " << j << " val: " << (*dest)[i].er 
		  << " .. (" << (*src)[j].er << " * " 
		  << srcmat [i + j * dest->size ()].weight << ")" << std::endl;
	*/
      }
    }
  }

  void LearnNet (int previous, int current) {
    real beta2 = 1e-08;
    real beta3 = beta2 * 1;
    //std::cout << "ALPHA: " << alpha << std::endl;
    //std::cout << "BETA: " << beta << std::endl;
    //std::cout << "BETA2: " << beta2 << std::endl;
    counter_++;

    if (current == -1)
      return;

    //Compute error vectors.  Note that, like ComputeNet, we are
    // only interested in words activated by the class for the 
    // current word, and NOT the entire vocabulary.
    int begin = v_.class_sizes_[v_.vocab_[current].class_index].begin;
    int end   = v_.class_sizes_[v_.vocab_[current].class_index].end;
    for (int i = begin; i <= end; i++) {
      neu2 [i].er = (0 - neu2 [i].ac);
      //std::cout << "er: " << neu2 [i].er << std::endl;
    }
    neu2 [current].er = (1 - neu2 [current].ac);

    //Flush error
    for (int i = 0; i < hsize_; i ++)
      neu1 [i].er = 0;
    for (int i = v_.vocab_.size (); i < osize_; i++)
      neu2 [i].er = (0 - neu2 [i].ac);

    neu2 [v_.vocab_ [current].class_index + v_.vocab_.size ()].er =
      (1 - neu2[v_.vocab_ [current].class_index + v_.vocab_.size ()].ac);
    /*
    std::cout << "Class error: " 
	      << neu2 [v_.vocab_[current].class_index + v_.vocab_.size ()].er
	      << std::endl;
    */
    //////////////////////////////////////
    //Begin learning direct connections for words
    if (ndirect_ > 0) {
      if (current != -1) {
	unsigned long long hash [MAX_NGRAM_ORDER];
	//Clear hash values
	for (int i = 0; i < direct_order; i++)
	  hash [i] = 0;

	for (int i = 0; i < direct_order; i++) {
	  if (i > 0)
	    if (history [i - 1] == -1)
	      break;
	  //Compute the hash
	  hash [i] = 
	    v_.primes_ [0] * v_.primes_ [1]
	    * (unsigned long long) (v_.vocab_ [current].class_index + 1);

	  for (int j = 1; j <= i; j++) {
	    hash [i] +=
	      v_.primes_[(i * v_.primes_[j] + j) % v_.primes_.size ()]
	      * (unsigned long long) (history [j - 1] + 1);
	  }
	  hash [i] = (hash [i] % (ndirect_ / 2)) + (ndirect_) / 2;
	}

	//Now update the direct connection synapse weights
	for (int i = begin; i <= end; i++) {
	  for (int j = 0; j < direct_order; j++) {
	    if (hash [j]) {
	      //std::cout << "Hash: " << j << " " << hash [j] << std::endl;
	      synd [hash [j]].weight +=
		alpha_ * neu2 [i].er - synd [hash [j]].weight * beta3;
	      hash [j]++;
	      hash [j] = hash [j] % ndirect_;
	    } else {
	      break;
	    }
	  }
	}
      }

      //Classes now
      unsigned long long hash [MAX_NGRAM_ORDER];
      //Clear all hash values
      for (int i = 0; i < direct_order; i++)
	hash [i] = 0;

      for (int i = 0; i < direct_order; i++) {
	if (i > 0)
	  if (history [i - 1] == -1)
	    break;
	hash [i] = v_.primes_[0] * v_.primes_[1];

	for (int j = 1; j <= i; j++) {
	  hash [i] += 
	    v_.primes_[(i * v_.primes_[j] + j) % v_.primes_.size ()]
	    * (unsigned long long) (history [j - 1] + 1);
	}
	hash [i] = hash [i] % (ndirect_ / 2);
      }

      for (int i = v_.vocab_.size (); i < osize_; i++) {
	for (int j = 0; j < direct_order; j++) {
	  if (hash [j]) {
	    synd [hash [j]].weight +=
	      alpha_ * neu2 [i].er - synd [hash [j]].weight * beta3;
	    hash [j]++;
	  } else {
	    break;
	  }
	}
      }
    }
    //End direct connection learning
    //////////////////////////////////////////////

    //Propagate basic error back through the network for the current timestep
    MatrixXVectorError (&neu2, &neu1, syn1, begin, end+1, 0, hsize_);

    //Take synapse index for the beginning of the class for the current word
    int t = begin * hsize_;
    for (int i = begin; i <= end; i++) {
      //Regularization anneals the weights
      if ((counter_ % 10) == 0) {
	for (int j = 0; j < hsize_; j++)
	  syn1 [j + t].weight += alpha_ * neu2 [i].er * neu1 [j].ac - syn1 [j + t].weight * beta2;
      } else {
	for (int j = 0; j < hsize_; j++) {
	  syn1 [j + t].weight += alpha_ * neu2 [i].er * neu1 [j].ac;
	  //std::cout << "syn1-new: " << syn1 [j + t].weight << std::endl;
	}
      }
      t += hsize_;
    }

    //Propagate errors for the class connections
    MatrixXVectorError (&neu2, &neu1, syn1, v_.vocab_.size (), osize_, 0, hsize_);
    //Weight updates for the class connections
    t = v_.vocab_.size () * hsize_;
    for (int i = v_.vocab_.size (); i < osize_; i++) {
      if ((counter_ % 10) == 0) {
	for (int j = 0; j < hsize_; j++) {
	  syn1 [j + t].weight += alpha_ * neu2 [i].er * neu1 [j].ac - syn1 [j+ t].weight * beta2;
	}
      } else {
	for (int j = 0; j < hsize_; j++) {
	  syn1 [j + t].weight += alpha_ * neu2 [i].er * neu1 [j].ac;
	  //std::cout << "syn1-newC: " << syn1 [j + t].weight << std::endl;
	}
      }
      t += hsize_;
    }

    //BPTT computation
    if (bptt <= 1) {
      /*
      for (int i = 0; i < hsize_; i++)
	neu1 [i].er = neu1 [i].er * neu1 [i].ac * (1 - neu1 [i].ac);

      if (previous != -1) {
	if ((counter_ % 10) == 0) {
      */
    } else {
      for (int i = 0; i < hsize_; i++) {
	bptt_hidden [i].ac = neu1 [i].ac;
	bptt_hidden [i].er = neu1 [i].er;
      }

      if (((counter_ % bptt_block) == 0) || current == 0) {
	//std::cout << "DO BPTT!" << std::endl;
	//Time step
	for (int step = 0; step < bptt + bptt_block - 2; step++) {
	  //Error derivation at hidden layer
	  for (int i = 0; i < hsize_; i++) {
	    neu1 [i].er = neu1 [i].er * neu1 [i].ac * (1 - neu1 [i].ac);
	  }

	  //Weight update 1->0
	  if (bptt_history [step] != -1) {
	    for (int i = 0; i < hsize_; i++) {
	      //isize_ = vocab_.size () + hsize_
	      //bptt_history [step] indexes into the synapse 
	      // array for the word ID for the nth word in the history
	      bptt_syn0 [bptt_history [step] + i * isize_].weight += 
		alpha_ * neu1 [i].er;
	    }
	  }

	  //Set the 'previous' hidden layer units to zero in the input layer
	  // this is prep for copying.
	  for (int i = isize_ - hsize_; i < isize_; i++) {
	    neu0 [i].er = 0;
	  }
	  //std::cout << "BPTT ERROR1: " << std::endl;
	  MatrixXVectorError (&neu1,
			      &neu0,
			      syn0,
			      0,
			      hsize_,
			      isize_ - hsize_,
			      isize_);

	  //Copy and update the synapses for the current hidden layer
	  for (int i = 0; i < hsize_; i++) {
	    for (int j = isize_ - hsize_; j < isize_; j++) {
	      bptt_syn0 [j + i * isize_].weight += alpha_ * neu1 [i].er * neu0 [j].ac;
	    }
	  }
	  //Propagate error from time T-n to T-n-1
	  for (int i = 0; i < hsize_; i++) {
	    neu1 [i].er = 
	      neu0 [i + isize_ - hsize_].er + bptt_hidden [(step + 1) * hsize_ + i].er;
	  }
	  //Propagate the activations from time T-n to T-n-1
	  if (step < bptt + bptt_block - 3) {
	    for (int i = 0; i < hsize_; i++) {
	      //Activation for hidden layer T-n-1
	      neu1 [i].ac = bptt_hidden [(step + 1) * hsize_ + i].ac;
	      //Activation for copied hidden layer in input = T-n-2
	      neu0 [i + isize_ - hsize_].ac =
		bptt_hidden [(step + 2) * hsize_ + i].ac;
	    }
	  }
	}
	//End first step-wise functions

	//Clear all the BPTT hidden-layer errors
	for (int i = 0; i < (bptt + bptt_block) * hsize_; i++) {
	  bptt_hidden [i].er = 0;
	}

	//Restore the current hidden layer activation after BPTT
	for (int i = 0; i < hsize_; i++) {
	  neu1 [i].ac = bptt_hidden [i].ac;
	}


	for (int i = 0; i < hsize_; i++) {
	  //Update the current synapse with the BPTT weight, then reset
	  // * This part handles the copied hidden layer
	  //Perform regularization at some incremental count
	  if ((counter_ % 10) == 0) {
	    for (int j = isize_ - hsize_; j < isize_; j++) {
	      syn0 [j + i * isize_].weight +=
		bptt_syn0 [j + i * isize_].weight - syn0 [j + i * isize_].weight * beta2;
	      bptt_syn0 [j + i * isize_].weight = 0;
	    }
	  } else {
	    for (int j = isize_ - hsize_; j < isize_; j++) {
	      syn0 [j + i * isize_].weight +=
		bptt_syn0 [j + i * isize_].weight;
	      bptt_syn0 [j + i * isize_].weight = 0;
	    }
	  }

	  //Step back through the bptt history / timesteps and
	  // * This part handles the word-history
	  //Perform regularization at some incremental count
	  if ((counter_ % 10) == 0) {
	    for (int step = 0; step < bptt + bptt_block - 2; step++) {
	      if (bptt_history [step] != -1) {
		syn0 [bptt_history [step] + i * isize_].weight +=
		  bptt_syn0 [bptt_history [step] + i * isize_].weight 
		  - syn0 [bptt_history [step] + i * isize_].weight 
		  * beta2;
		bptt_syn0 [bptt_history [step] + i * isize_].weight = 0;
	      }
	    }
	  } else {
	    for (int step = 0; step < bptt + bptt_block - 2; step++) {
	      if (bptt_history [step] != -1) {
		syn0 [bptt_history [step] + i * isize_].weight +=
		  bptt_syn0 [bptt_history [step] + i * isize_].weight;
		bptt_syn0 [bptt_history [step] + i * isize_].weight = 0;
	      }
	    }
	  }
	}
	/////////////////////////
      }
    }  
  }

  void ClearActivations () {
    for (int i = 0; i < v_.vocab_.size (); i++)
      neu0 [i].Clear ();
    for (int i = v_.vocab_.size (); i < isize_; i++) 
      neu0 [i].Clear (0.1);
    for (int i = 0; i < hsize_; i++)
      neu1 [i].Clear ();
    for (int i = 0; i < osize_; i++)
      neu2 [i].Clear ();
  }

  double EvaluateSentence (const vector<int>& sentence) {
    ClearActivations ();
    NetReset ();
    for (int i = 0; i < MAX_NGRAM_ORDER; i++)
      history [i] = 0;
    
    int prev = 0;
    double prob = 0.0;
    
    for (int i = 0; i < sentence.size (); i++) {
      prob += log10 (ComputeNet (prev, sentence[i]));
      CopyHiddenLayerToInput ();
      if (prev != -1)
	neu0 [prev].ac = 0;
      prev = sentence [i];
      for (int j = MAX_NGRAM_ORDER - 1; j > 0; j--)
	history [j] = history [j - 1];
      history [0] = prev;
    }
    ClearActivations ();
    NetReset ();
    return prob;
  }

  void CopyHiddenLayerToInput () {
    for (int i = 0; i < hsize_; i++)
      neu0 [i + isize_ - hsize_].ac = neu1 [i].ac;
  }

  void NetReset () {
    for (int i = 0; i < hsize_; i++) {
      neu1 [i].ac = 1.0;
    }

    CopyHiddenLayerToInput ();
    
    if (bptt > 0) {
      for (int i = 1; i < bptt + bptt_block; i++) {
	bptt_history [i] = 0;
      }

      for (int i = bptt + bptt_block - 1; i > 1; i--) {
	for (int j = 0; j < hsize_; j++) {
	  bptt_hidden [i * hsize_ + j].ac = 0;
	  bptt_hidden [i * hsize_ + j].er = 0;
	}
      }
    }

    for (int i = 0; i < MAX_NGRAM_ORDER; i++) {
      history [i] = 0;
    }
  }

  void WriteRnnLMModel (const string& ofilename) {
    //Write out the model in RnnLM text format.
    std::ofstream ofile;
    ofile.open (ofilename);
    ofile << "version: 10" << endl;
    ofile << "file format: 0" << endl;
    ofile << endl;
    ofile << "training data file: " << endl;
    ofile << "validation data file: " << endl;
    ofile << endl;
    ofile << "last probability of validation data: 0.000000" << endl;
    ofile << "number of finished iterations: 4" << endl;
    ofile << "current position in training data: 0" << endl;
    ofile << "current probability of training data: 0.000000" << endl;
    ofile << "save after processing # words: 0" << endl;
    ofile << "# of training words: 0" << endl;
    ofile << "input layer size: " << isize_ << endl;
    ofile << "hidden layer size: " << hsize_ << endl;
    ofile << "compression layer size: 0" << endl;
    ofile << "output layer size: " << osize_ << endl;
    ofile << "direct connections: " << ndirect_ << endl;
    ofile << "direct order: " << direct_order << endl;
    ofile << "bptt: " << bptt << endl;
    ofile << "bptt block: " << bptt_block << endl;
    ofile << "vocabulary size: " << v_.vocab_.size () << endl;
    ofile << "class size: " << v_.class_size_ << endl;
    ofile << "old classes: 0" << endl;
    ofile << "independent sentences mode: 1" << endl;
    ofile << "starting learning rate: 0.100000" << endl;
    ofile << "current learning rate: 0.100000" << endl;
    ofile << "learning rate decrease: 0" << endl;
    ofile << "\n" << endl;
    ofile << "Vocabulary:" << endl;
    for (int i = 0; i < v_.vocab_.size (); i++) {
      ofile << std::setfill (' ') << std::setw (6) << i;
      ofile << std::setw (11) << v_.vocab_[i].cn;
      ofile << "\t" << v_.vocab_[i].word;
      ofile << "   \t" << v_.vocab_[i].class_index << endl;
    }
    ofile << endl;
    ofile << std::setprecision (4) << std::fixed;
    ofile << "Hidden layer activation:" << endl;
    for (int i = 0; i < hsize_; i++)
      ofile << "1.0000" << endl;
    ofile << endl;
    ofile << "Weights 0->1:" << endl;
    for (int i = 0; i < syn0.size (); i++)
      ofile << syn0 [i].weight << endl;
    ofile << "\n" << endl;
    ofile << "Weights 1->2:" << endl;
    for (int i = 0; i < syn1.size (); i++)
      ofile << syn1 [i].weight << endl;
    ofile << endl;
    ofile << "Direct connections:" << endl;
    ofile << std::setprecision (2);
    for (int i = 0; i < synd.size (); i++)
      ofile << synd [i].weight << endl;
    ofile.close ();
  }

  const LegacyRnnLMHash& v_;
  int hsize_;
  int ndirect_;
  int bptt;
  int bptt_block;
  int direct_order;
  int isize_;
  int osize_;
  int counter_;
  real alpha_;
  real beta_;
  std::vector<Neuron> neu0;  // Input layer
  std::vector<Neuron> neu1;  // Hidden layer
  std::vector<Neuron> neu2;  // Output layer
  std::vector<Synapse> syn0; // Connections from Input->Hidden
  std::vector<Synapse> syn1; // Connections from Hidden->Output
  std::vector<Synapse> synd; // Direct connections
  std::vector<int> bptt_history;
  std::vector<Neuron> bptt_hidden;
  std::vector<Synapse> bptt_syn0;
  std::vector<int> history;
};


#endif // BRNNLM_H__
