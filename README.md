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
# by AltFstAligner or phonetisaurus-align.  m2m-aligner may also be used, 
# but the formatting will need to be fixed. [the below is an example,
#  which assumes that phonetisaurus-align is available, the dict is not
#  provided in this distribution, but the examples from phonetisaurus may
#  be used.
$ cd script/
$ phonetisaurus-align --intput=test.dict --ofile=test.corpus --seq1_del=false

#This will train up an rnnlm with some reasonable G2P parameters
# There are two stages to this: 
#  1. A model is trained in the standard fashion, using 90% training data, 10% validation.
#  2. A second model is trained using the Alpha values from the first training stage, then 
#     this is used to manually train the model - including the validation data.
# See --help and the example output for details.  You may need to tweak these a bit!
$  ./train-g2p-rnnlm.py -c test.corpus -p test
rnnlm -train test.train -valid test.valid -rnnlm test.rnnlm    -independent -binary -bptt 6 -bptt-block 10    -direct 15 -direct-order 5 -hidden 110 -class 45
rnnlm -one-iter -train test.corpus -alpha 0.100000 -rnnlm x.m.rnnlm    -independent -binary -bptt 6 -bptt-block 10    -direct 15 -direct-order 5 -hidden 110 -class 45
rnnlm -one-iter -train test.corpus -alpha 0.100000 -rnnlm x.m.rnnlm    -independent -binary -bptt 6 -bptt-block 10    -direct 15 -direct-order 5 -hidden 110 -class 45
rnnlm -one-iter -train test.corpus -alpha 0.100000 -rnnlm x.m.rnnlm    -independent -binary -bptt 6 -bptt-block 10    -direct 15 -direct-order 5 -hidden 110 -class 45
rnnlm -one-iter -train test.corpus -alpha 0.100000 -rnnlm x.m.rnnlm    -independent -binary -bptt 6 -bptt-block 10    -direct 15 -direct-order 5 -hidden 110 -class 45
rnnlm -one-iter -train test.corpus -alpha 0.100000 -rnnlm x.m.rnnlm    -independent -binary -bptt 6 -bptt-block 10    -direct 15 -direct-order 5 -hidden 110 -class 45
rnnlm -one-iter -train test.corpus -alpha 0.100000 -rnnlm x.m.rnnlm    -independent -binary -bptt 6 -bptt-block 10    -direct 15 -direct-order 5 -hidden 110 -class 45
rnnlm -one-iter -train test.corpus -alpha 0.050000 -rnnlm x.m.rnnlm    -independent -binary -bptt 6 -bptt-block 10    -direct 15 -direct-order 5 -hidden 110 -class 45
rnnlm -one-iter -train test.corpus -alpha 0.025000 -rnnlm x.m.rnnlm    -independent -binary -bptt 6 -bptt-block 10    -direct 15 -direct-order 5 -hidden 110 -class 45
rnnlm -one-iter -train test.corpus -alpha 0.012500 -rnnlm x.m.rnnlm    -independent -binary -bptt 6 -bptt-block 10    -direct 15 -direct-order 5 -hidden 110 -class 45
rnnlm -one-iter -train test.corpus -alpha 0.006250 -rnnlm x.m.rnnlm    -independent -binary -bptt 6 -bptt-block 10    -direct 15 -direct-order 5 -hidden 110 -class 45

# Run the model with the toy corpus 
# Provide a file instead of 'echo' if you have a file) [original model]
$ ../phonetisaurus-g2prnn --rnnlm=test.rnnlm --test=<(echo "TEST") --nbest=5 | prettify.pl
TEST			  T EH S T	  12.0902
TEST			  T IY S T	  17.346
TEST			  T EY S T	  17.6871
TEST			  T EH S 17.7112
TEST			  T AH S T	18.8496
# And using the retrained model (.m.rnnlm)
$ ../phonetisaurus-g2prnn --rnnlm=test.m.rnnlm --test=<(echo "TEST") --nbest=5 | prettify.pl
TEST				     T EH S T		     11.6143
TEST				     T IY S T		     16.9946
TEST				     T EY S T		     17.3594
TEST				     T S T  17.3626
TEST				     T AH S T	17.37
...

#See --help for more arguments.  If running a large test set, note that the g2prnn
# decoder has OpenMP support.  You can set the number of parallel threads with the
# --threads=N parameter.
#If you are NOT using the default delimeter '}', you need to specify this to the 
# decoder AND to the 'prettify' script, e.g.:
$ ../phonetisaurus-g2prnn --rnnlm=ru.rnnlm --gpdelim="#" \
  --test=<(echo "английский") --nbest=1 | ./prettify.pl false "#"
английский		  a n g l i y s k i y		 17.2149
...
```


EXAMPLE MODELS:
================
Two 'reasonable' example models for the CMUdict can be downloaded here:
#### Forward:
 * https://www.dropbox.com/s/60hqp1irs4hq9u2/g014b2b.rnnlm

#### Backward:
 * https://www.dropbox.com/s/rf7r9m2vhkvxyju/g014b2b.rev.rnnlm
