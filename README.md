RnnLMG2P
========

Hacking about with joint sequence RnnLMs

First 'reasonably' fast version.  Direct decoding with a joint-sequence RnnLM.
This uses a representation of the RnnLM that is a bit more efficient than the 
default for the purposes of decoding.

Synapse arrays are shared.
The size of the input array is the same size as the hidden-layer.
The output layer is not stored anywhere, but rather allocated when ComputeNet is called.

The current copy of the hidden layer activations and associated word-history
is stored in the search tokens.

The BRnnLM.h is the beginning of a full reimplementation of RnnLM with a mind to 
customizing it for G2P.  Main features currently under development:
  * Phoneme category features for histories
  * Future-features for graphemes
  * Generic class definitions
  * Full Bi-directional implementation

This will be folded back into Phonetisaurus after I make a bit more progress.

This also addresses several issues with the default RnnLM where G2P is concerned:
  * Shuffling of training data
  * BPTT issues related to -independent option
  * Model compression.  The neurons and much of the direct-connections synapse 
     table can probably be compressed a good deal.  These are mostly empty.

USAGE:
================
```
$ cd src/
$ make && make install
$ cd ..
$ ./phonetisaurus-g2prnn --rnnlm=g014b2b.rnnlm --test=script/spaced.100.wlist \
  --beam=6 --nbest=1 | ./script/prettify.pl
ABANDONED     AH B AE N D AH N D </s>	   17.2936
ABBATIELLO    AA B AA T IY EH L OW </s>	   18.0768
ABBENHAUS     AE B AH N HH AW S </s>	   21.0564
ABBEVILLE     AE B V IH L </s>	15.0214
ABBOUD	      AH B AW D </s>	14.2237
ABBREVIATION  AH B R IY V IY EY SH AH N </s>	12.2586
ABDOLLAH      AE B D AA L AH </s>  16.4175
ABDUCTEES     AE B D AH K T IY Z </s>	13.7622
...
```

EXAMPLE MODELS:
================
A 'reasonable' example model for the CMUdict can be downloaded here:

 * https://www.dropbox.com/s/60hqp1irs4hq9u2/g014b2b.rnnlm

