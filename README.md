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

Quickstart USAGE:
================
```
$ cd src/
$ make && make install
$ cd ..
#Train up a model - probably best to use a smallish corpus to start!
#--corpus should be an aligned joint-token corpus such as that output
# by AltFstAligner or phonetisaurus-align.
$ cd script/
#This will train up an rnnlm with some reasonable G2P parameters
# See --help and the example output for details.  You may need to tweak these a bit!
$ ./train-g2p-rnnlm.py --corpus test.corpus

# Run the model with the toy corpus (provide a file instead of 'echo' if you have a file)
$ ../phonetisaurus-g2prnn --rnnlm=test.rnnlm --test=<(echo "PERSISTANT") --nbest=5 \
  | ./prettify.pl
PERSISTANT P ER S IH S T AH N T	24.258
PERSISTANT P ER S AH S T AE N T	27.4669
PERSISTANT P ER S AH S T AA N T	27.5301
PERSISTANT P ER S IH S AH N T 27.5923
PERSISTANT P ER S AH S T EY N T	29.8902
...

#See --help for more arguments.  If running a large test set, note that the g2prnn
# decoder has OpenMP support.  You can set the number of parallel threads with the
# --threads=N parameter.
#If you are NOT using the default delimeter '}', you need to specify this to the decoder
# AND to the 'prettify' script, e.g.:
$ ../phonetisaurus-g2prnn --rnnlm=ru.rnnlm --gpdelim="#" \
  --test=<(echo "английский") --nbest=1 | ./prettify.pl false "#"
английский		  a n g l i y s k i y		 17.2149
...
```


EXAMPLE MODELS:
================
A 'reasonable' example model for the CMUdict can be downloaded here:

 * https://www.dropbox.com/s/60hqp1irs4hq9u2/g014b2b.rnnlm

