#!/usr/bin/python
import re, random, os, sys

def ShuffleTrainingCorpus (ifilename, ofile_prefix, pvalid=0.1) :
    """
      Shuffle the training corpus, then split into training
      and validation partitions based on the pvalid value.
    """
    
    corpus = []
    for line in open (ifilename, "r") :
        corpus.append (line)

    random.shuffle (corpus)

    ofp_train = open ("{0}.train".format (ofile_prefix), "w")
    ofp_valid = open ("{0}.valid".format (ofile_prefix), "w")

    num_valid = int (pvalid * len (corpus))
    for i, entry in enumerate (corpus) :
        if i < num_valid :
            ofp_valid.write (entry)
        else :
            ofp_train.write (entry)
    ofp_train.close ()
    ofp_valid.close ()

    return

def TrainRnnLM (args) :
    train_file = "{0}.train".format (args.prefix)
    valid_file = "{0}.valid".format (args.prefix)
    rnnlm_file = "{0}.rnnlm".format (args.prefix)
    if os.path.isfile (rnnlm_file) == True :
        print "rnnlm file: {0} already exists!".format (rnnlm_file)
        print "Quitting because rnnlm tool will update this instead of retraining."
        print "Please choose a new prefix or delete the old rnnlm."
        sys.exit ()
    command = "./rnnlm -train {0} -valid {1} -rnnlm {2} \
   -independent -binary -bptt {3} -bptt-block {4} \
   -direct {5} -direct-order {6} -hidden {7} -class {8} &> {9}.log"
    command = command.format (train_file, valid_file, rnnlm_file,
                              args.bptt, args.bptt_block, 
                              args.direct, args.direct_order,
                              args.hidden, args.classes, rnnlm_file)
    print command
    os.system (command)
    return

def ReadRnnLMLog (args) :
    logfile = args.logfile.format (args.prefix)
    alphas = []
    for line in open (logfile, "r") :
        if line.startswith ("Iter:") :
            line = re.sub (r"^.*Alpha: ", "", line.strip ())
            alphas.append (line)

    return alphas

def TrainRnnLMManually (args, alphas) :
    rnnlm_file = "{0}.m.rnnlm".format (args.prefix)

    for alpha in alphas :
        command = "./rnnlm -one-iter -train {0} -alpha {1} -rnnlm {2} \
   -independent -binary -bptt {3} -bptt-block {4} \
   -direct {5} -direct-order {6} -hidden {7} -class {8} &> {9}.log"
        command = command.format (args.corpus, alpha, rnnlm_file,
                                  args.bptt, args.bptt_block, 
                                  args.direct, args.direct_order,
                                  args.hidden, args.classes, rnnlm_file)
        print command
        os.system (command)
    return
    
if __name__ == "__main__" :
    import sys, argparse

    example = "{0} --corpus test.corpus".format (sys.argv [0])
    parser  = argparse.ArgumentParser (description = example)
    parser.add_argument ("--corpus", "-c", help="Input aligned corpus for training.", required=True)
    parser.add_argument ("--prefix", "-p", help="Output file prefix. Used for .train, .valid, .rnnlm.", default="test")
    parser.add_argument ("--bptt", "-b", help="Number of BPTT steps.", default=6, type=int)
    parser.add_argument ("--bptt_block", "-bb", help="Increment at which to propagate error.", default=10, type=int)
    parser.add_argument ("--direct", "-d", help="Number of direct connections (in millions)", default=15, type=int)
    parser.add_argument ("--direct_order", "-do", help="Maximum n-gram order for direct connections.", default=5, type=int)
    parser.add_argument ("--classes", "-nc", help="Number of classes.", default=45, type=int)
    parser.add_argument ("--hidden", "-nh", help="Number of nodes in the hidden layer.", default=110, type=int)
    parser.add_argument ("--pvalid", "-pv", help="Percent of corpus to use as validation data.", default=0.1, type=float)
    parser.add_argument ("--logfile", "-l", help="Use an existing logfile to manually train a model.", default="{0}.rnnlm.log")
    parser.add_argument ("--verbose", "-v", help="Verbose mode.", default=False, action="store_true")
    args = parser.parse_args ()

    if args.verbose :
        for k,v in args.__dict__.iteritems () :
            print k, "->", v

    #Splite the corpus into test/valid
    ShuffleTrainingCorpus (args.corpus, args.prefix, args.pvalid)
    #Train the initial RnnLM - find the set of alpha values
    TrainRnnLM (args)
    #Extract the alpha values from the logfile
    alphas = ReadRnnLMLog (args)
    #Retrain the RnnLM using *all* data, manually, with the alpha vals
    # learned in the previous step
    TrainRnnLMManually (args, alphas) 
