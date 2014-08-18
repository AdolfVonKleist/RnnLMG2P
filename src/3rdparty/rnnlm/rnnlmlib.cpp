///////////////////////////////////////////////////////////////////////
//
// Recurrent neural network based statistical language modeling toolkit
// Version 0.3e
// (c) 2010-2012 Tomas Mikolov (tmikolov@gmail.com)
//
///////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "rnnlmlib.h"

///// fast exp() implementation
static union{
    double d;
    struct{
        int j,i;
        } n;
} d2i;
#define EXP_A (1048576/M_LN2)
#define EXP_C 60801
#define FAST_EXP(y)(d2i.n.i=EXP_A*(y)+(1072693248-EXP_C),d2i.d)

///// include blas
#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif
//


real CRnnLM::random(real min, real max)
{
    return rand()/(real)RAND_MAX*(max-min)+min;
}

void CRnnLM::setTrainFile(char *str)
{
    strcpy(train_file, str);
}

void CRnnLM::setValidFile(char *str)
{
    strcpy(valid_file, str);
}

void CRnnLM::setTestFile(char *str)
{
    strcpy(test_file, str);
}

void CRnnLM::setRnnLMFile(char *str)
{
    strcpy(rnnlm_file, str);
}




void CRnnLM::readWord(char *word, FILE *fin)
{
    int a=0, ch;

    while (!feof(fin)) {
      ch = fgetc(fin);

      if (ch == 13) continue;

      if ((ch == ' ') || (ch=='\t') || (ch=='\n')) {
	if (a > 0) {
	  if (ch == '\n') ungetc (ch, fin);
	  break;
	}

	if (ch == '\n') {
	  strcpy(word, (char *)"</s>");
	  return;
	}
	else continue;
      }

      word[a] = ch;
      a++;

      if (a >= MAX_STRING) {
	//printf("Too long word found!\n");   //truncate too long words
	a--;
      }
    }
    word[a]=0;
}

int CRnnLM::getWordHash (char *word)
{
    unsigned int hash, a;
    
    hash = 0;
    for (a = 0; a < strlen (word); a++) 
      hash = hash * 237 + word[a];
    hash = hash % vocab_hash_size;
    
    return hash;
}

int CRnnLM::searchVocab (char *word)
{
    int a;
    unsigned int hash;
    
    hash = getWordHash (word);
    
    if (vocab_hash[hash]==-1) 
      return -1;
    if (!strcmp (word, vocab[vocab_hash[hash]].word)) 
      return vocab_hash[hash];
    
    for (a=0; a<vocab_size; a++) {				//search in vocabulary
      if (!strcmp(word, vocab[a].word)) {
	vocab_hash[hash]=a;
	return a;
      }
    }

    return -1;							//return OOV if not found
}

int CRnnLM::readWordIndex(FILE *fin)
{
    char word[MAX_STRING];

    readWord(word, fin);
    if (feof(fin)) return -1;

    return searchVocab(word);
}

int CRnnLM::addWordToVocab(char *word)
{
    unsigned int hash;
    
    strcpy(vocab[vocab_size].word, word);
    vocab[vocab_size].cn=0;
    vocab_size++;

    if (vocab_size+2>=vocab_max_size) {        //reallocate memory if needed
        vocab_max_size+=100;
        vocab=(struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
    }
    
    hash=getWordHash(word);
    vocab_hash[hash]=vocab_size-1;

    return vocab_size-1;
}

void CRnnLM::sortVocab()
{
    int a, b, max;
    vocab_word swap;
    
    for (a=1; a<vocab_size; a++) {
        max=a;
        for (b=a+1; b<vocab_size; b++) if (vocab[max].cn<vocab[b].cn) max=b;

        swap=vocab[max];
        vocab[max]=vocab[a];
        vocab[a]=swap;
    }
}

void CRnnLM::learnVocabFromTrainFile()    //assumes that vocabulary is empty
{
    char word[MAX_STRING];
    FILE *fin;
    int a, i, train_wcn;
    
    for (a=0; a<vocab_hash_size; a++) vocab_hash[a]=-1;

    fin=fopen(train_file, "rb");

    vocab_size=0;

    addWordToVocab((char *)"</s>");

    train_wcn=0;
    while (1) {
        readWord(word, fin);
        if (feof(fin)) break;
        
        train_wcn++;

        i=searchVocab(word);
        if (i==-1) {
            a=addWordToVocab(word);
            vocab[a].cn=1;
        } else vocab[i].cn++;
    }

    sortVocab();
    
    //select vocabulary size
    /*a=0;
    while (a<vocab_size) {
	a++;
	if (vocab[a].cn==0) break;
    }
    vocab_size=a;*/

    if (debug_mode>0) {
	printf("Vocab size: %d\n", vocab_size);
	printf("Words in train file: %d\n", train_wcn);
    }
    
    train_words=train_wcn;

    fclose(fin);
}

void CRnnLM::saveWeights()      //saves current weights and unit activations
{
    int a,b;

    for (a=0; a<layer0_size; a++) {
        neu0b[a].ac=neu0[a].ac;
        neu0b[a].er=neu0[a].er;
    }

    for (a=0; a<layer1_size; a++) {
        neu1b[a].ac=neu1[a].ac;
        neu1b[a].er=neu1[a].er;
    }
    
    for (a=0; a<layerc_size; a++) {
        neucb[a].ac=neuc[a].ac;
        neucb[a].er=neuc[a].er;
    }
    
    for (a=0; a<layer2_size; a++) {
        neu2b[a].ac=neu2[a].ac;
        neu2b[a].er=neu2[a].er;
    }
    
    for (b=0; b<layer1_size; b++) for (a=0; a<layer0_size; a++) {
	syn0b[a+b*layer0_size].weight=syn0[a+b*layer0_size].weight;
    }
    
    if (layerc_size>0) {
	for (b=0; b<layerc_size; b++) for (a=0; a<layer1_size; a++) {
	    syn1b[a+b*layer1_size].weight=syn1[a+b*layer1_size].weight;
	}
	
	for (b=0; b<layer2_size; b++) for (a=0; a<layerc_size; a++) {
	    syncb[a+b*layerc_size].weight=sync[a+b*layerc_size].weight;
	}
    }
    else {
	for (b=0; b<layer2_size; b++) for (a=0; a<layer1_size; a++) {
	    syn1b[a+b*layer1_size].weight=syn1[a+b*layer1_size].weight;
	}
    }
    
    //for (a=0; a<direct_size; a++) syn_db[a].weight=syn_d[a].weight;
}

void CRnnLM::restoreWeights()      //restores current weights and unit activations from backup copy
{
    int a,b;

    for (a=0; a<layer0_size; a++) {
        neu0[a].ac=neu0b[a].ac;
        neu0[a].er=neu0b[a].er;
    }

    for (a=0; a<layer1_size; a++) {
        neu1[a].ac=neu1b[a].ac;
        neu1[a].er=neu1b[a].er;
    }
    
    for (a=0; a<layerc_size; a++) {
        neuc[a].ac=neucb[a].ac;
        neuc[a].er=neucb[a].er;
    }
    
    for (a=0; a<layer2_size; a++) {
        neu2[a].ac=neu2b[a].ac;
        neu2[a].er=neu2b[a].er;
    }

    for (b=0; b<layer1_size; b++) for (a=0; a<layer0_size; a++) {
        syn0[a+b*layer0_size].weight=syn0b[a+b*layer0_size].weight;
    }
    
    if (layerc_size>0) {
	for (b=0; b<layerc_size; b++) for (a=0; a<layer1_size; a++) {
	    syn1[a+b*layer1_size].weight=syn1b[a+b*layer1_size].weight;
	}
	
	for (b=0; b<layer2_size; b++) for (a=0; a<layerc_size; a++) {
	    sync[a+b*layerc_size].weight=syncb[a+b*layerc_size].weight;
	}
    }
    else {
	for (b=0; b<layer2_size; b++) for (a=0; a<layer1_size; a++) {
	    syn1[a+b*layer1_size].weight=syn1b[a+b*layer1_size].weight;
	}
    }
    
    //for (a=0; a<direct_size; a++) syn_d[a].weight=syn_db[a].weight;
}

void CRnnLM::saveContext()		//useful for n-best list processing
{
    int a;
    
    for (a=0; a<layer1_size; a++) neu1b[a].ac=neu1[a].ac;
}

void CRnnLM::restoreContext()
{
    int a;
    
    for (a=0; a<layer1_size; a++) neu1[a].ac=neu1b[a].ac;
}

void CRnnLM::saveContext2()
{
    int a;
    
    for (a=0; a<layer1_size; a++) neu1b2[a].ac=neu1[a].ac;
}

void CRnnLM::restoreContext2()
{
    int a;
    
    for (a=0; a<layer1_size; a++) neu1[a].ac=neu1b2[a].ac;
}

/*
void CRnnLM::SaveContext (std::string& id) {
  for (int a = 0; a < layer1_size; a++)
    NeuMap[id].neu[a].ac = neu1[a].ac;
}

void CRnnLM::RestoreContext (std::string& id) {
  for (int a = 0; a < layer1_size; a++)
    neu1[a].ac = NeuMap[id].neu[a].ac;
}
*/
void CRnnLM::initNet()
{
    int a, b, cl;

    layer0_size=vocab_size+layer1_size;
    layer2_size=vocab_size+class_size;

    neu0=(struct neuron *)calloc(layer0_size, sizeof(struct neuron));
    neu1=(struct neuron *)calloc(layer1_size, sizeof(struct neuron));
    neuc=(struct neuron *)calloc(layerc_size, sizeof(struct neuron));
    neu2=(struct neuron *)calloc(layer2_size, sizeof(struct neuron));

    syn0=(struct synapse *)calloc(layer0_size*layer1_size, sizeof(struct synapse));
    if (layerc_size==0)
	syn1=(struct synapse *)calloc(layer1_size*layer2_size, sizeof(struct synapse));
    else {
	syn1=(struct synapse *)calloc(layer1_size*layerc_size, sizeof(struct synapse));
	sync=(struct synapse *)calloc(layerc_size*layer2_size, sizeof(struct synapse));
    }

    if (syn1==NULL) {
	printf("Memory allocation failed\n");
	exit(1);
    }
    
    if (layerc_size>0) if (sync==NULL) {
	printf("Memory allocation failed\n");
	exit(1);
    }
    
    syn_d=(direct_t *)calloc((long long)direct_size, sizeof(direct_t));

    if (syn_d==NULL) {
	printf("Memory allocation for direct connections failed (requested %lld bytes)\n", (long long)direct_size * (long long)sizeof(direct_t));
	exit(1);
    }

    neu0b=(struct neuron *)calloc(layer0_size, sizeof(struct neuron));
    neu1b=(struct neuron *)calloc(layer1_size, sizeof(struct neuron));
    neucb=(struct neuron *)calloc(layerc_size, sizeof(struct neuron));
    neu1b2=(struct neuron *)calloc(layer1_size, sizeof(struct neuron));
    neu2b=(struct neuron *)calloc(layer2_size, sizeof(struct neuron));

    syn0b=(struct synapse *)calloc(layer0_size*layer1_size, sizeof(struct synapse));
    //syn1b=(struct synapse *)calloc(layer1_size*layer2_size, sizeof(struct synapse));
    if (layerc_size==0)
	syn1b=(struct synapse *)calloc(layer1_size*layer2_size, sizeof(struct synapse));
    else {
	syn1b=(struct synapse *)calloc(layer1_size*layerc_size, sizeof(struct synapse));
	syncb=(struct synapse *)calloc(layerc_size*layer2_size, sizeof(struct synapse));
    }

    if (syn1b==NULL) {
	printf("Memory allocation failed\n");
	exit(1);
    }
    
    for (a=0; a<layer0_size; a++) {
        neu0[a].ac=0;
        neu0[a].er=0;
    }

    for (a=0; a<layer1_size; a++) {
        neu1[a].ac=0;
        neu1[a].er=0;
    }
    
    for (a=0; a<layerc_size; a++) {
        neuc[a].ac=0;
        neuc[a].er=0;
    }
    
    for (a=0; a<layer2_size; a++) {
        neu2[a].ac=0;
        neu2[a].er=0;
    }

    //cout << "Syn0:" << endl;
    for (b=0; b<layer1_size; b++) for (a=0; a<layer0_size; a++) {
	real weight = random (-0.1, 0.1) + random (-0.1, 0.1) + random (-0.1, 0.1);
	//cout << "i: " << a << " j: " << b << " val: " << weight << endl;
        syn0[a + b * layer0_size].weight = weight;
    }

    if (layerc_size>0) {
	for (b=0; b<layerc_size; b++) for (a=0; a<layer1_size; a++) {
	    syn1[a+b*layer1_size].weight=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
	}
	
	for (b=0; b<layer2_size; b++) for (a=0; a<layerc_size; a++) {
	    sync[a+b*layerc_size].weight=random(-0.1, 0.1)+random(-0.1, 0.1)+random(-0.1, 0.1);
	}
    }
    else {
      //cout << "Syn1: " << endl;
      for (b=0; b<layer2_size; b++) 
	for (a=0; a<layer1_size; a++) {
	  real weight = random (-0.1, 0.1) + random (-0.1, 0.1) + random (-0.1, 0.1);
	  //cout << "i: " << a << " j: " << b << " val: " << weight << endl;
	  syn1[a + b * layer1_size].weight = weight;
	}
    }
    
    long long aa;
    for (aa=0; aa<direct_size; aa++) syn_d[aa]=0;
    
    if (bptt>0) {
	bptt_history=(int *)calloc((bptt+bptt_block+10), sizeof(int));
	for (a=0; a<bptt+bptt_block; a++) bptt_history[a]=-1;
	//
	bptt_hidden=(neuron *)calloc((bptt+bptt_block+1)*layer1_size, sizeof(neuron));
	for (a=0; a<(bptt+bptt_block)*layer1_size; a++) {
	    bptt_hidden[a].ac=0;
	    bptt_hidden[a].er=0;
	}
	//
	bptt_syn0=(struct synapse *)calloc(layer0_size*layer1_size, sizeof(struct synapse));
	if (bptt_syn0==NULL) {
	    printf("Memory allocation failed\n");
	    exit(1);
	}
    }

    saveWeights();
    
    double df, dd;
    int i;
    
    df=0;
    dd=0;
    a=0;
    b=0;

    if (old_classes) {  	// old classes
        for (i=0; i<vocab_size; i++) b+=vocab[i].cn;
        for (i=0; i<vocab_size; i++) {
            df+=vocab[i].cn/(double)b;
            if (df>1) df=1;
            if (df>(a+1)/(double)class_size) {
    	        vocab[i].class_index=a;
    	        if (a<class_size-1) a++;
            } else {
    	        vocab[i].class_index=a;
            }
        }
    } else {			// new classes
        for (i=0; i<vocab_size; i++) b+=vocab[i].cn;
        for (i=0; i<vocab_size; i++) dd+=sqrt(vocab[i].cn/(double)b);
        for (i=0; i<vocab_size; i++) {
	    df+=sqrt(vocab[i].cn/(double)b)/dd;
            if (df>1) df=1;
            if (df>(a+1)/(double)class_size) {
    	        vocab[i].class_index=a;
    	        if (a<class_size-1) a++;
            } else {
    	        vocab[i].class_index=a;
            }
	}
    }
    
    //allocate auxiliary class variables (for faster search when normalizing probability at output layer)
    
    class_words=(int **)calloc(class_size, sizeof(int *));
    class_cn=(int *)calloc(class_size, sizeof(int));
    class_max_cn=(int *)calloc(class_size, sizeof(int));
    
    for (i=0; i<class_size; i++) {
	class_cn[i]=0;
	class_max_cn[i]=10;
	class_words[i]=(int *)calloc(class_max_cn[i], sizeof(int));
    }
    
    for (i=0; i<vocab_size; i++) {
	cl=vocab[i].class_index;
	class_words[cl][class_cn[cl]]=i;
	class_cn[cl]++;
	if (class_cn[cl]+2>=class_max_cn[cl]) {
	    class_max_cn[cl]+=10;
	    class_words[cl]=(int *)realloc(class_words[cl], class_max_cn[cl]*sizeof(int));
	}
    }
}

void CRnnLM::saveNet()       //will save the whole network structure                                                        
{
    FILE *fo;
    int a, b;
    char str[1000];
    float fl;
    
    sprintf(str, "%s.temp", rnnlm_file);

    fo=fopen(str, "wb");
    if (fo==NULL) {
        printf("Cannot create file %s\n", rnnlm_file);
        exit(1);
    }
    fprintf(fo, "version: %d\n", version);
    fprintf(fo, "file format: %d\n\n", filetype);

    fprintf(fo, "training data file: %s\n", train_file);
    fprintf(fo, "validation data file: %s\n\n", valid_file);

    fprintf(fo, "last probability of validation data: %f\n", llogp);
    fprintf(fo, "number of finished iterations: %d\n", iter);

    fprintf(fo, "current position in training data: %d\n", train_cur_pos);
    fprintf(fo, "current probability of training data: %f\n", logp);
    fprintf(fo, "save after processing # words: %d\n", anti_k);
    fprintf(fo, "# of training words: %d\n", train_words);

    fprintf(fo, "input layer size: %d\n", layer0_size);
    fprintf(fo, "hidden layer size: %d\n", layer1_size);
    fprintf(fo, "compression layer size: %d\n", layerc_size);
    fprintf(fo, "output layer size: %d\n", layer2_size);

    fprintf(fo, "direct connections: %lld\n", direct_size);
    fprintf(fo, "direct order: %d\n", direct_order);
    
    fprintf(fo, "bptt: %d\n", bptt);
    fprintf(fo, "bptt block: %d\n", bptt_block);
    
    fprintf(fo, "vocabulary size: %d\n", vocab_size);
    fprintf(fo, "class size: %d\n", class_size);
    
    fprintf(fo, "old classes: %d\n", old_classes);
    fprintf(fo, "independent sentences mode: %d\n", independent);
    
    fprintf(fo, "starting learning rate: %f\n", starting_alpha);
    fprintf(fo, "current learning rate: %f\n", alpha);
    fprintf(fo, "learning rate decrease: %d\n", alpha_divide);
    fprintf(fo, "\n");

    fprintf(fo, "\nVocabulary:\n");
    for (a=0; a<vocab_size; a++) fprintf(fo, "%6d\t%10d\t%s\t%d\n", a, vocab[a].cn, vocab[a].word, vocab[a].class_index);

    
    if (filetype==TEXT) {
	fprintf(fo, "\nHidden layer activation:\n");
	for (a=0; a<layer1_size; a++) fprintf(fo, "%.4f\n", neu1[a].ac);
    }
    if (filetype==BINARY) {
    	for (a=0; a<layer1_size; a++) {
    	    fl=neu1[a].ac;
    	    fwrite(&fl, 4, 1, fo);
    	}
    }
    //////////
    if (filetype==TEXT) {
	fprintf(fo, "\nWeights 0->1:\n");
	for (b=0; b<layer1_size; b++) {
    	    for (a=0; a<layer0_size; a++) {
        	fprintf(fo, "%.4f\n", syn0[a+b*layer0_size].weight);
    	    }
	}
    }
    if (filetype==BINARY) {
	for (b=0; b<layer1_size; b++) {
    	    for (a=0; a<layer0_size; a++) {
    		fl=syn0[a+b*layer0_size].weight;
    		fwrite(&fl, 4, 1, fo);
    	    }
	}
    }
    /////////
    if (filetype==TEXT) {
	if (layerc_size>0) {
	    fprintf(fo, "\n\nWeights 1->c:\n");
	    for (b=0; b<layerc_size; b++) {
		for (a=0; a<layer1_size; a++) {
    		    fprintf(fo, "%.4f\n", syn1[a+b*layer1_size].weight);
    		}
    	    }
    	
    	    fprintf(fo, "\n\nWeights c->2:\n");
	    for (b=0; b<layer2_size; b++) {
		for (a=0; a<layerc_size; a++) {
    		    fprintf(fo, "%.4f\n", sync[a+b*layerc_size].weight);
    		}
    	    }
	}
	else
	{
	    fprintf(fo, "\n\nWeights 1->2:\n");
	    for (b=0; b<layer2_size; b++) {
		for (a=0; a<layer1_size; a++) {
    		    fprintf(fo, "%.4f\n", syn1[a+b*layer1_size].weight);
    		}
    	    }
    	}
    }
    if (filetype==BINARY) {
	if (layerc_size>0) {
	    for (b=0; b<layerc_size; b++) {
		for (a=0; a<layer1_size; a++) {
		    fl=syn1[a+b*layer1_size].weight;
    		    fwrite(&fl, 4, 1, fo);
    		}
    	    }
    	
	    for (b=0; b<layer2_size; b++) {
		for (a=0; a<layerc_size; a++) {
    		    fl=sync[a+b*layerc_size].weight;
    		    fwrite(&fl, 4, 1, fo);
    		}
    	    }
	}
	else
	{
	    for (b=0; b<layer2_size; b++) {
		for (a=0; a<layer1_size; a++) {
    		    fl=syn1[a+b*layer1_size].weight;
    		    fwrite(&fl, 4, 1, fo);
    		}
    	    }
    	}
    }
    ////////
    if (filetype==TEXT) {
	fprintf(fo, "\nDirect connections:\n");
	long long aa;
	for (aa=0; aa<direct_size; aa++) {
    	    fprintf(fo, "%.2f\n", syn_d[aa]);
	}
    }
    if (filetype==BINARY) {
	long long aa;
	for (aa=0; aa<direct_size; aa++) {
    	    fl=syn_d[aa];
    	    fwrite(&fl, 4, 1, fo);
    	    
    	    /*fl=syn_d[aa]*4*256;			//saving direct connections this way will save 50% disk space; several times more compression is doable by clustering
    	    if (fl>(1<<15)-1) fl=(1<<15)-1;
    	    if (fl<-(1<<15)) fl=-(1<<15);
    	    si=(signed short int)fl;
    	    fwrite(&si, 2, 1, fo);*/
	}
    }
    ////////    
    fclose(fo);
    
    rename(str, rnnlm_file);
}

void CRnnLM::goToDelimiter(int delim, FILE *fi)
{
    int ch=0;

    while (ch!=delim) {
        ch=fgetc(fi);
        if (feof(fi)) {
            printf("Unexpected end of file\n");
            exit(1);
        }
    }
}

/*
void CRnnLM::MapJointToken (vocab_word* word) {
  string delim = "}";
  string tmp   = word->word;
  token_map[searchVocab (word->word)] = tmp;

  vector<string> parts = tokenize_utf8_string (&tmp, &delim);
  if (parts.size() == 2) {
    if (joint_map.find (parts[0]) == joint_map.end()) {
      vector<int> tokens;
      tokens.push_back (searchVocab (word->word));
      joint_map[parts[0]] = tokens;
    } else {
      joint_map[parts[0]].push_back (searchVocab (word->word));
    }
  } else {
    vector<int> tokens;
    tokens.push_back (searchVocab (word->word));
    joint_map[tmp] = tokens;
  }
}

vector<int>& CRnnLM::SearchJointVocab (string& word) {
  return joint_map [word];
}
*/

void CRnnLM::restoreNet()    //will read whole network structure
{
    FILE *fi;
    int a, b, ver;
    float fl;
    char str[MAX_STRING];
    double d;

    fi=fopen(rnnlm_file, "rb");
    if (fi==NULL) {
	printf("ERROR: model file '%s' not found!\n", rnnlm_file);
	exit(1);
    }

    goToDelimiter(':', fi);
    fscanf(fi, "%d", &ver);
    if ((ver==4) && (version==5)) /* we will solve this later.. */ ; else
    if (ver!=version) {
        printf("Unknown version of file %s\n", rnnlm_file);
        printf("ver: %s  version: %s\n", ver, version);
        exit(1);
    }
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &filetype);
    //
    goToDelimiter(':', fi);
    if (train_file_set==0) {
	fscanf(fi, "%s", train_file);
    } else fscanf(fi, "%s", str);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%s", valid_file);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &llogp);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &iter);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &train_cur_pos);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &logp);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &anti_k);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &train_words);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer0_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer1_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layerc_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &layer2_size);
    //
    if (ver>5) {
	goToDelimiter(':', fi);
	fscanf(fi, "%lld", &direct_size);
    }
    //
    if (ver>6) {
	goToDelimiter(':', fi);
	fscanf(fi, "%d", &direct_order);
    }
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &bptt);
    //
    if (ver>4) {
	goToDelimiter(':', fi);
	fscanf(fi, "%d", &bptt_block);
    } else bptt_block=10;
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &vocab_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &class_size);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &old_classes);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &independent);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%lf", &d);
    starting_alpha=d;
    //
    goToDelimiter(':', fi);
    if (alpha_set==0) {
	fscanf(fi, "%lf", &d);
	alpha=d;
    } else fscanf(fi, "%lf", &d);
    //
    goToDelimiter(':', fi);
    fscanf(fi, "%d", &alpha_divide);
    //
    
    
    //read normal vocabulary
    if (vocab_max_size<vocab_size) {
	if (vocab!=NULL) 
	  free(vocab);
        vocab_max_size=vocab_size+1000;
        vocab = (struct vocab_word *) 
	  calloc (vocab_max_size, sizeof(struct vocab_word));    
        //initialize memory for vocabulary
    }

    goToDelimiter(':', fi);
    for (a=0; a<vocab_size; a++) {
	//fscanf(fi, "%d%d%s%d", &b, &vocab[a].cn, vocab[a].word, &vocab[a].class_index);
	fscanf(fi, "%d%d", &b, &vocab[a].cn);
	readWord(vocab[a].word, fi);
	/*
	if (joint == true)
	  MapJointToken (&vocab[a]);
	*/
	fscanf(fi, "%d", &vocab[a].class_index);
	//printf("%d  %d  %s  %d\n", b, vocab[a].cn, vocab[a].word, vocab[a].class_index);
    }

    //
    if (neu0==NULL) initNet();		//memory allocation here
    //
    
    
    if (filetype==TEXT) {
	goToDelimiter(':', fi);
	for (a=0; a<layer1_size; a++) {
	    fscanf(fi, "%lf", &d);
	    neu1[a].ac=d;
	}
    }
    if (filetype==BINARY) {
	fgetc(fi);
	for (a=0; a<layer1_size; a++) {
	    fread(&fl, 4, 1, fi);
	    neu1[a].ac=fl;
	}
    }
    //
    if (filetype==TEXT) {
	goToDelimiter(':', fi);
	//cout << "Syn0 orig: " << endl;
	for (b=0; b<layer1_size; b++) {
    	    for (a=0; a<layer0_size; a++) {
		fscanf(fi, "%lf", &d);
		//cout << "b: " << b << " a: " << a << " val: " << d << endl;
		syn0[a+b*layer0_size].weight=d;
	    }
	}
    }
    if (filetype==BINARY) {
	for (b=0; b<layer1_size; b++) {
    	    for (a=0; a<layer0_size; a++) {
    		fread(&fl, 4, 1, fi);
		syn0[a+b*layer0_size].weight=fl;
    	    }
	}
    }
    //
    if (filetype==TEXT) {
	goToDelimiter(':', fi);
	//cout << "Syn1 orig: " << endl;
	if (layerc_size==0) {	//no compress layer
	    for (b=0; b<layer2_size; b++) {
    		for (a=0; a<layer1_size; a++) {
		    fscanf(fi, "%lf", &d);
		    //cout << "b: " << b << " a: " << a << " val: " << d << endl;
		    syn1[a+b*layer1_size].weight=d;
		}
    	    }
	}
	else
	{				//with compress layer
	    for (b=0; b<layerc_size; b++) {
    		for (a=0; a<layer1_size; a++) {
		    fscanf(fi, "%lf", &d);
		    syn1[a+b*layer1_size].weight=d;
		}
    	    }
    	
    	    goToDelimiter(':', fi);
    	
    	    for (b=0; b<layer2_size; b++) {
    		for (a=0; a<layerc_size; a++) {
		    fscanf(fi, "%lf", &d);
		    sync[a+b*layerc_size].weight=d;
		}
    	    }
	}
    }
    if (filetype==BINARY) {
	if (layerc_size==0) {	//no compress layer
	    for (b=0; b<layer2_size; b++) {
    		for (a=0; a<layer1_size; a++) {
    		    fread(&fl, 4, 1, fi);
		    syn1[a+b*layer1_size].weight=fl;
    		}
    	    }
	}
	else
	{				//with compress layer
	    for (b=0; b<layerc_size; b++) {
    		for (a=0; a<layer1_size; a++) {
    		    fread(&fl, 4, 1, fi);
		    syn1[a+b*layer1_size].weight=fl;
    		}
    	    }
    	
    	    for (b=0; b<layer2_size; b++) {
    		for (a=0; a<layerc_size; a++) {
    		    fread(&fl, 4, 1, fi);
		    sync[a+b*layerc_size].weight=fl;
    		}
    	    }
	}
    }
    //
    if (filetype==TEXT) {
	goToDelimiter(':', fi);		//direct conenctions
	long long aa;
    	for (aa=0; aa<direct_size; aa++) {
	    fscanf(fi, "%lf", &d);
	    syn_d[aa]=d;
	}
    }
    //
    if (filetype==BINARY) {
	long long aa;
    	for (aa=0; aa<direct_size; aa++) {
    	    fread(&fl, 4, 1, fi);
	    syn_d[aa]=fl;
	    
	    /*fread(&si, 2, 1, fi);
	    fl=si/(float)(4*256);
	    syn_d[aa]=fl;*/
    	}
    }
    //
    
    saveWeights();

    fclose(fi);
}

void CRnnLM::netFlush()   //cleans all activations and error vectors
{
    int a;

    for (a=0; a<layer0_size-layer1_size; a++) {
        neu0[a].ac=0;
        neu0[a].er=0;
    }

    for (a=layer0_size-layer1_size; a<layer0_size; a++) {   //last hidden layer is initialized to vector of 0.1 values to prevent unstability
        neu0[a].ac=0.1;
        neu0[a].er=0;
    }

    for (a=0; a<layer1_size; a++) {
        neu1[a].ac=0;
        neu1[a].er=0;
    }
    
    for (a=0; a<layerc_size; a++) {
        neuc[a].ac=0;
        neuc[a].er=0;
    }
    
    for (a=0; a<layer2_size; a++) {
        neu2[a].ac=0;
        neu2[a].er=0;
    }
}

void CRnnLM::netReset()   //cleans hidden layer activation + bptt history
{
    int a, b;

    for (a = 0; a < layer1_size; a++) {
        neu1 [a].ac = 1.0;
    }

    copyHiddenLayerToInput ();

    if (bptt > 0) {
        for (a = 1; a < bptt + bptt_block; a++) 
	  bptt_history [a] = 0;
        for (a = bptt + bptt_block - 1; a > 1; a--) {
	  for (b = 0; b < layer1_size; b++) {
            bptt_hidden [a * layer1_size + b].ac = 0;
            bptt_hidden [a * layer1_size + b].er = 0;
	  }
	}
    }

    for (a = 0; a < MAX_NGRAM_ORDER; a++) 
      history [a] = 0;
}

void CRnnLM::matrixXvector (struct neuron *dest, struct neuron *srcvec, struct synapse *srcmatrix, int matrix_width, int from, int to, int from2, int to2, int type) {
    int a, b;
    real val1, val2, val3, val4;
    real val5, val6, val7, val8;
    //cout << from << ", " << to << ", " << from2 << ", " << to2 << endl;
    //cout.precision (10);
    if (type == 0) {		//ac mod
      //cout << "MVals:" << endl;
      for (b = from; b < to; b++) {
	for (a = from2; a < to2; a++) {
	  //cout << "BEFORE: " << fixed << dest[b].ac;
	  dest[b].ac += srcvec[a].ac * srcmatrix[a + b * matrix_width].weight;
	  //cout << " ACTIVATE: i: " << a << " j: " << b << " val: " << dest[b].ac << " .. " 
	  //     << "(" << srcvec[a].ac << " * " << srcmatrix[a + b * matrix_width].weight 
	  //     << ")" << endl;
	}
      }
    } else { 		//er mod
      if (type==1) {
	for (a=from2; a<to2; a++) {
	  for (b=from; b<to; b++) {
	    dest[a].er += srcvec[b].er * srcmatrix[a+b*matrix_width].weight;
            /*
	    cout << "ERROR: i: " << a << " j: " << b << " val: " << dest[a].er << " .. " 
	         << "(" << srcvec[b].er << " * " << srcmatrix[a + b * matrix_width].weight 
	         << ")" << endl;
	    */
	  }
    	}
      }
    }
}


void CRnnLM::computeNet (int last_word, int word) {
    //Sum is used for normalization: it's better to have larger 
    // precision as many numbers are summed together here
    int a, b, c;
    real val;
    double sum;   
    
    if (last_word != -1) 
      neu0[last_word].ac = 1;

    //propagate 0->1
    //First set everything to zero
    //cout << "Input layer activations:" << endl;
    for (a = 0; a < layer1_size; a++) {
      neu1[a].ac = 0;
      //cout << a << ": " << neu1[a].ac << " ";
    }
   //cout << endl;

    //for (a = 0; a < layerc_size; a++) 
    //  neuc[a].ac = 0;
    matrixXvector (neu1, neu0, syn0, layer0_size, 0, layer1_size, layer0_size - layer1_size, layer0_size, 0);
    
    //cout << "Hidden layer activations:" << endl;
    for (b = 0; b < layer1_size; b++) {
      a = last_word;
      //Update the Net value for each hidden unit with the information 
      // from the previous word
      if (a != -1) 
	neu1[b].ac += neu0[a].ac * syn0 [a + b * layer0_size].weight;
    }
    
    //activate 1      --sigmoid
    // Uses *all* information: words + previous layer hidden activations
    for (a = 0; a < layer1_size; a++) {
      if (neu1[a].ac > 50) 
	neu1[a].ac = 50;  //for numerical stability
      if (neu1[a].ac < -50) 
	neu1[a].ac = -50;  //for numerical stability
      val = -neu1[a].ac;
      //This is just the normal sigmoid function
      // Activate the result of the augmented summation
      neu1[a].ac = 1 / (1 + FAST_EXP (val));
      //cout << a << ": " << neu1[a].ac << " ";
    }
    //cout << endl;
    
    /*
    if (layerc_size>0) {
	matrixXvector(neuc, neu1, syn1, layer1_size, 0, layerc_size, 0, layer1_size, 0);
	//activate compression      --sigmoid
	for (a=0; a<layerc_size; a++) {
	    if (neuc[a].ac>50) neuc[a].ac=50;  //for numerical stability
    	    if (neuc[a].ac<-50) neuc[a].ac=-50;  //for numerical stability
    	    val=-neuc[a].ac;
    	    neuc[a].ac=1/(1+FAST_EXP(val));
	}
    }
    */
        
    //1->2 class
    for (b = vocab_size; b < layer2_size; b++) 
      neu2[b].ac = 0;
    
    //if (layerc_size > 0) {
    //matrixXvector (neu2, neuc, sync, layerc_size, vocab_size, layer2_size, 0, layerc_size, 0);
    //} else {
    matrixXvector (neu2, neu1, syn1, layer1_size, vocab_size, layer2_size, 0, layer1_size, 0);
	//}

    /////////////////////////////////////////////////
    //Begin direct connection activations for classes
    if (direct_size > 0) {
        //This will hold pointers to syn_d that contains hash parameters
      unsigned long long hash[MAX_NGRAM_ORDER];	
	
      for (a = 0; a < direct_order; a++) 
	hash [a] = 0;
	
      for (a = 0; a < direct_order; a++) {
	b = 0;
	//If OOV was in history, do not use this N-gram feature and higher orders
	if (a > 0) 
	  if (history [a - 1] == -1) 
	    break;	
	hash [a] = PRIMES [0] * PRIMES [1];
	    	    
	for (b = 1; b <= a; b++) {
          //Update hash value based on words from the history
	  hash [a] += 
	    PRIMES [(a * PRIMES [b] + b) % PRIMES_SIZE] 
	    * (unsigned long long) (history [b - 1] + 1);	
	}
        //Make sure that starting hash index is in the first 
	// half of syn_d (second part is reserved for history->words features)
	hash [a] = hash [a] % (direct_size / 2);		
      }
	
      for (a = vocab_size; a < layer2_size; a++) {
	for (b = 0; b < direct_order; b++) {
	  if (hash [b]) {
            //Apply current parameter and move to the next one.  
	    // Note: 'a' index should be in the output layer class space
	    neu2 [a].ac += syn_d [hash [b]];		
	    hash[b]++;
	  } else { 
	    break;
	  }
	}
      }
    }
    //End direct connection activations for classes
    ///////////////////////////////////////////////
    
    
    //Activation 2   --softmax on classes
    sum = 0;
    for (a = vocab_size; a < layer2_size; a++) {
	if (neu2[a].ac > 50) 
	  neu2[a].ac = 50;  //for numerical stability
	if (neu2[a].ac < -50) 
	  neu2[a].ac = -50;  //for numerical stability
        val = FAST_EXP (neu2[a].ac);
        sum += val;
        neu2[a].ac = val;
    }
    for (a = vocab_size; a < layer2_size; a++)  {
      neu2[a].ac /= sum;         //output layer activations now sum exactly to 1
      //cout << "oclase: " << a << " " << neu2 [a].ac << endl;
    }
    //1->2 word
    //Perform the 'net' computation for the words: hidden layer to output
    if (word != -1) {
      //Here we activate all words in the CLASS for the current word
      for (c = 0; c < class_cn [vocab [word].class_index]; c++) 
	neu2 [class_words [vocab [word].class_index] [c]].ac = 0;
      //cout << "1->2 words: Class: " << vocab [word].class_index << endl;
      //cout << "Begin: " << class_words[vocab[word].class_index][0] << " End: " << class_words[vocab[word].class_index][0] + class_cn[vocab[word].class_index] << endl;
      matrixXvector (neu2, 
		     neu1, 
		     syn1, 
		     layer1_size, 
		     class_words[vocab[word].class_index][0], 
		     class_words[vocab[word].class_index][0] + class_cn[vocab[word].class_index], 
		     0, 
		     layer1_size, 
		     0);
      //cout << "Done" << endl;
    }
    
    //Begin direct connection activations for words
    if (word != -1) {
      if (direct_size > 0) {
	unsigned long long hash [MAX_NGRAM_ORDER];
	//Reset the hash pointers
	for (a = 0; a < direct_order; a++) 
	  hash [a] = 0;
	
	//Find hash activations for ngram order
	for (a = 0; a < direct_order; a++) {
	  b = 0;
	  if (a > 0) {
	    //Quit if we don't have a previous word
	    if (history [a - 1] == -1) {
	      break;
	    }
	  }
	  //Hash the current word-class index, make sure it isn't zero
	  hash [a] = 
	    PRIMES [0] * PRIMES[1] 
	    * (unsigned long long) (vocab [word].class_index + 1);
	  //Increment the class-hash with history words up to the 
	  // current order
	  for (b = 1; b <= a; b++) {
	    hash[a] += 
	      PRIMES [(a * PRIMES [b] + b) % PRIMES_SIZE]
	      * (unsigned long long) (history [b - 1] + 1);
	  }
	  //Ensure that the resulting hash index falls into the
	  // second half of the syn_d synapse vector
	  hash [a] = (hash [a] % (direct_size / 2)) + (direct_size / 2);
	}
	
	//Iterate through all associated words for the current class
	for (c = 0; c < class_cn [vocab [word].class_index]; c++) {
	  //Output layer word index for word 'c' in current class
	  a = class_words [vocab [word].class_index] [c];
	  //Note: 'b' indexes into the word-space of the syn_d synapse
	  // vector.  Increment neu2 activation. All direct connections
	  // are updated and hash is updated.  I still don't get it.
	  for (b = 0; b < direct_order; b++) {
	    if (hash [b]) {
	      //cout << "Hash: " << b << " " << hash [b] << endl;
	      neu2 [a].ac += syn_d [hash [b]];
	      hash [b]++;
	      hash [b] = hash [b] % direct_size;
	    } else {
	      break;
	    }
	  }
	}
      }
    }
    //End direct connection activations for words
    /////////////////////////////////////////////


    //Activation 2   --softmax on words
    sum = 0;
    if (word != -1) {
      //Again, all words in the class for the current word
      for (c = 0; c < class_cn [vocab [word].class_index]; c++) {
	a = class_words [vocab [word].class_index] [c];
	if (neu2[a].ac > 50) 
	  neu2[a].ac = 50;  //for numerical stability
	if (neu2[a].ac < -50) 
	  neu2[a].ac = -50;  //for numerical stability
	val = FAST_EXP (neu2[a].ac);
	sum += val;
	neu2[a].ac = val;
      }
      //Now normalize the word outputs
      for (c = 0; c < class_cn [vocab [word].class_index]; c++) {
	neu2 [class_words [vocab [word].class_index] [c]].ac /= sum;
	//std::cout << "Activated: " << c << " : " 
	//	  << neu2 [class_words [vocab [word].class_index] [c]].ac << std::endl;
      }
    }
    /*
    cout << "Current: " << word << ", Prev: " << last_word << "\t";
    cout << " P (w|C): " << neu2[word].ac
	 << " P (C): " << neu2[vocab_size + vocab [word].class_index].ac 
	 << " P (w) = " << neu2[word].ac * neu2[vocab_size + vocab [word].class_index].ac
	 << endl;
    */
}

void CRnnLM::learnNet(int last_word, int word)
{
    int a, b, c, t, step;
    real beta2, beta3;

    beta2 = 1e-08; //beta * alpha;
    beta3 = beta2 * 1;	
    //Beta3 can be possibly larger than beta2, 
    // as that is useful on small datasets 
    // (if the final model is to be interpolated wich 
    // backoff model) - todo in the future

    //cout << "ALPHA: " << alpha << endl;
    //cout << "BETA: " << beta << endl;
    //cout << "BETA2: " << beta2 << endl;
    if (word == -1) return;

    //Compute error vectors
    for (c = 0; c < class_cn [vocab [word].class_index]; c++) {
	a = class_words [vocab [word].class_index][c];
        neu2[a].er = (0 - neu2[a].ac);
	//cout << "er: " << neu2[a].er << endl;
    }
    neu2 [word].er = (1 - neu2 [word].ac);	//word part

    //flush error
    for (a = 0; a < layer1_size; a++) 
      neu1[a].er = 0;
    for (a = 0; a < layerc_size; a++) 
      neuc[a].er = 0;

    for (a = vocab_size; a < layer2_size; a++) {
      neu2 [a].er = (0 - neu2 [a].ac);
    }

    neu2 [vocab [word].class_index + vocab_size].er = 
      (1 - neu2 [vocab [word].class_index + vocab_size].ac);	//class part
    //cout << "Class error: " << neu2 [vocab [word].class_index + vocab_size].er << endl;
    
    //
    //Learn direct connections between words
    //Why is this done in REVERSE order to the activation?
    //Because gradient descent?
    if (direct_size > 0) {	
      if (word != -1) {
	unsigned long long hash [MAX_NGRAM_ORDER];
	//First, just find the hash value
	for (a = 0; a < direct_order; a++) 
	  hash [a] = 0;
	
	for (a = 0; a < direct_order; a++) {
	  b = 0;
	  if (a > 0) 
	    if (history [a - 1] == -1) 
	      break;
	  //Hash the class index
	  hash [a] = 
	    PRIMES [0] * PRIMES [1] 
	    * (unsigned long long) (vocab [word].class_index + 1);
				
	  //Update the hash with the history.  In effect this creates
	  // a hash of the N-gram history.
	  for (b = 1; b <= a; b++) 
	    hash [a] += 
	      PRIMES [(a * PRIMES [b] + b) % PRIMES_SIZE] 
	      * (unsigned long long)(history [b - 1] + 1);
	  hash [a] = (hash [a] % (direct_size / 2)) + (direct_size) / 2;
	}
	
	//Now update the direct connection synapse weights
	for (c = 0; c < class_cn [vocab [word].class_index]; c++) {
	  a = class_words [vocab [word].class_index] [c];
	    
	  for (b = 0; b < direct_order; b++) {
	    if (hash [b]) {
	      //cout << "Hash: " << b << " " << hash [b] << endl;
	      //Update with the error for current word 'a' from the 
	      // alpha-scaled, current activated class, regularized by the annealed
	      // current weight value for syn_d.
	      //Why do we do l2 regularization at every step?
	      syn_d [hash [b]] +=
		alpha * neu2[a].er - syn_d [hash [b]] * beta3;
	      hash[b]++;
	      hash [b] = hash [b] % direct_size;
	    } else {
	      break;
	    }
	  }
	}
      }
    }

    //
    //learn direct connections to classes
    if (direct_size > 0) {	//learn direct connections between words and classes
      unsigned long long hash[MAX_NGRAM_ORDER];
	
      for (a = 0; a < direct_order; a++) 
	hash[a] = 0;
	
      for (a = 0; a < direct_order; a++) {
	b = 0;
	if (a > 0) 
	  if (history [a - 1] == -1) 
	    break;
	hash [a] = PRIMES [0] * PRIMES [1];
	    	    
	for (b = 1; b <= a; b++) 
	  hash [a] += 
	    PRIMES [(a * PRIMES [b] + b) % PRIMES_SIZE]
	    * (unsigned long long) (history [b - 1] + 1);
	hash [a] = hash [a] % (direct_size / 2);
      }
	
      for (a = vocab_size; a < layer2_size; a++) {
	for (b = 0; b < direct_order; b++) {
	  if (hash [b]) {
	    syn_d [hash [b]] += 
	      alpha * neu2[a].er - syn_d [hash [b]] * beta3;
	    hash[b]++;
	  } else {
	    break;
	  }
	}
      }
    }
    //


    //Propagate basic error back through the network for the current timestep
    {
    matrixXvector (neu1, 
		   neu2, 
		   syn1, 
		   layer1_size, 
		   class_words [vocab [word].class_index][0], 
		   class_words[vocab[word].class_index][0] + class_cn[vocab[word].class_index], 
		   0, 
		   layer1_size, 
		   1);
    	
    t = class_words [vocab [word].class_index][0] * layer1_size;
    for (c = 0; c < class_cn [vocab [word].class_index]; c++) {
      b = class_words [vocab [word].class_index][c];
      if ((counter % 10) == 0)	{//regularization is done every 10. step
	for (a = 0; a < layer1_size; a++) {
	  syn1 [a + t].weight += alpha * neu2 [b].er * neu1 [a].ac - syn1 [a + t].weight * beta2;
	}
      } else {
	for (a = 0; a < layer1_size; a++) {
	  syn1 [a + t].weight += alpha * neu2 [b].er * neu1[a].ac;
	  //cout << "syn1-new: " << syn1 [a + t].weight << endl;
	}
      }
      t += layer1_size;
    }
	//
    matrixXvector (neu1, 
		   neu2, 
		   syn1, 
		   layer1_size, 
		   vocab_size, 
		   layer2_size, 
		   0, 
		   layer1_size, 
		   1);		//propagates errors 2->1 for classes
	
    c = vocab_size * layer1_size;
    for (b = vocab_size; b < layer2_size; b++) {
      if ((counter % 10) == 0) {	//regularization is done every 10. step
	for (a = 0; a < layer1_size; a++) 
	  syn1 [a + c].weight += alpha * neu2 [b].er * neu1[a].ac - syn1 [a + c].weight * beta2;	//weight 1->2 update
      }	else {
	for (a = 0; a < layer1_size; a++)  {
	  syn1 [a + c].weight += alpha * neu2[b].er * neu1[a].ac;	//weight 1->2 update
	  //cout << "syn1-newC: " << syn1 [a + c].weight << endl;
	}
      }
      c += layer1_size;
    }
    }
    
    //
    
    ///////////////
    
    if (bptt <= 1) {		//bptt==1 -> normal BP
      for (a = 0; a < layer1_size; a++) //error derivation at layer 1
	neu1 [a].er = neu1 [a].er * neu1 [a].ac * (1 - neu1 [a].ac);    

	//weight update 1->0
      a = last_word;
      if (a != -1) {
	if ((counter % 10) == 0) {
	  for (b = 0; b < layer1_size; b++) {
	    syn0 [a + b * layer0_size].weight += 
	      alpha * neu1 [b].er * neu0 [a].ac - syn0 [a + b * layer0_size].weight * beta2;
	  }
	} else {
	  for (b = 0; b < layer1_size; b++) {
	    syn0 [a + b * layer0_size].weight += 
	      alpha * neu1 [b].er * neu0[a].ac;
	  }
	}

	if ((counter % 10) == 0) {
	  for (b = 0; b < layer1_size; b++) {
	    for (a = layer0_size - layer1_size; a < layer0_size; a++) {
	      syn0 [a + b * layer0_size].weight += 
		alpha * neu1 [b].er * neu0 [a].ac - syn0 [ a + b * layer0_size].weight * beta2;
	    }
	  }
	} else {
	  for (b = 0; b < layer1_size; b++) { 
	    for (a = layer0_size - layer1_size; a < layer0_size; a++) {
	      syn0 [a + b * layer0_size].weight += 
		alpha * neu1 [b].er * neu0 [a].ac;
	    }
	  }
	}
      }
	//BPTT
    } else { 
      for (b = 0; b < layer1_size; b++) 
	bptt_hidden [b].ac = neu1 [b].ac;
      for (b = 0; b < layer1_size; b++) 
	bptt_hidden [b].er = neu1 [b].er;
	
      if (((counter % bptt_block) == 0) || (independent && (word == 0))) {
	//cout << "DO BPTT!" << endl;
	for (step = 0; step < bptt + bptt_block - 2; step++) {
	  for (a = 0; a < layer1_size; a++) 
	    neu1[a].er = neu1[a].er * neu1[a].ac * (1 - neu1[a].ac);    //error derivation at layer 1
	    
	  //weight update 1->0
	  a = bptt_history [step];
	  if (a != -1)
	    for (b = 0; b < layer1_size; b++) {
	      //*neu0[a].ac; --should be always set to 1
	      bptt_syn0 [a + b * layer0_size].weight += alpha * neu1 [b].er;
	    }
	    
	  for (a = layer0_size - layer1_size; a < layer0_size; a++) 
	    neu0[a].er = 0;
	  //cout << "BPTT ERROR1: " << endl;
	  matrixXvector (neu0, 
			 neu1, 
			 syn0, 
			 layer0_size, 
			 0, 
			 layer1_size, 
			 layer0_size-layer1_size, 
			 layer0_size, 
			 1);		//propagates errors 1->0

	  for (b = 0; b < layer1_size; b++) 
	    for (a = layer0_size - layer1_size; a < layer0_size; a++) {
	      //neu0[a].er += neu1[b].er * syn0[a+b*layer0_size].weight;
	      bptt_syn0 [a + b * layer0_size].weight += alpha * neu1 [b].er * neu0 [a].ac;
	    }
	    
	  // Propagate error from time T-n to T-n-1
	  for (a = 0; a < layer1_size; a++) {
	    neu1 [a].er = 
	      neu0 [a + layer0_size - layer1_size].er + bptt_hidden [(step + 1) * layer1_size + a].er;
	  }
	    
	  if (step < bptt + bptt_block - 3)
	    for (a = 0; a < layer1_size; a++) {
	      neu1 [a].ac = bptt_hidden [(step + 1) * layer1_size + a].ac;
	      neu0 [a + layer0_size - layer1_size].ac = 
		bptt_hidden [(step + 2) * layer1_size + a].ac;
	    }
	}
	    
	for (a = 0; a < (bptt + bptt_block) * layer1_size; a++) {
	  bptt_hidden[a].er = 0;
	}
	
	for (b = 0; b < layer1_size; b++) 
	  neu1 [b].ac = bptt_hidden [b].ac;		//restore hidden layer after bptt
	
	    //
	for (b = 0; b < layer1_size; b++) {		//copy temporary syn0
	  if ((counter % 10) == 0) {
	    for (a = layer0_size - layer1_size; a < layer0_size; a++) {
	      syn0 [a + b * layer0_size].weight += 
		bptt_syn0 [a + b * layer0_size].weight - syn0 [a + b * layer0_size].weight * beta2;
	      bptt_syn0 [a + b * layer0_size].weight = 0;
	    }
	  } else {
	    for (a = layer0_size - layer1_size; a < layer0_size; a++) {
	      syn0 [a + b * layer0_size].weight +=
		bptt_syn0 [a + b * layer0_size].weight;
	      bptt_syn0 [a + b * layer0_size].weight = 0;
	    }
	  }
	    
	  if ((counter % 10) == 0) {
	    for (step = 0; step < bptt + bptt_block - 2; step++) 
	      if (bptt_history [step] != -1) {
		syn0 [bptt_history [step] + b * layer0_size].weight += 
		  bptt_syn0 [bptt_history [step] + b * layer0_size].weight 
		  - syn0 [bptt_history [step] + b * layer0_size].weight 
		  * beta2;
		bptt_syn0 [bptt_history [step] + b * layer0_size].weight = 0;
	      }
	  } else {
	    for (step = 0; step < bptt + bptt_block - 2; step++) 
	      if (bptt_history [step] !=- 1) {
		syn0 [bptt_history [step] + b * layer0_size].weight +=
		  bptt_syn0 [bptt_history [step] + b * layer0_size].weight;
		bptt_syn0 [bptt_history [step] + b * layer0_size].weight = 0;
	      }
	  }
	}
      }
    }	
}

void CRnnLM::copyHiddenLayerToInput()
{
    int a;

    for (a=0; a<layer1_size; a++) {
        neu0[a+layer0_size-layer1_size].ac=neu1[a].ac;
    }
}

void CRnnLM::trainNet()
{
    int a, b, word, last_word, wordcn;
    char log_name[200];
    FILE *fi, *flog;
    clock_t start, now;

    sprintf(log_name, "%s.output.txt", rnnlm_file);

    printf("Starting training using file %s\n", train_file);
    starting_alpha=alpha;
    
    fi=fopen(rnnlm_file, "rb");
    if (fi!=NULL) {
	fclose(fi);
	printf("Restoring network from file to continue training...\n");
	restoreNet();
    } else {
	learnVocabFromTrainFile();
	initNet();
	iter=0;
    }

    if (class_size>vocab_size) {
	printf("WARNING: number of classes exceeds vocabulary size!\n");
    }
    
    counter=train_cur_pos;
    
    //saveNet();

    while (1) {
        printf("Iter: %3d\tAlpha: %f\n   ", iter, alpha);
        fflush(stdout);
        //cout << "BPTT: " << bptt << endl;
        if (bptt > 0) 
	  for (a = 0; a < bptt + bptt_block; a++) 
	    bptt_history[a] = 0;

        for (a = 0; a < MAX_NGRAM_ORDER; a++) 
	  history[a] = 0;

        //TRAINING PHASE
        netFlush ();

        fi = fopen (train_file, "rb");
        last_word = 0;

        start = clock();
        
        while (1) {
	  //cout << "Counter: " << counter << endl;
    	    counter++;
	    /*
	    for (int i = MAX_NGRAM_ORDER - 1; i >= 0; i--)
	      cout << history [i] << " " << vocab [history[i]].word << ", ";
	    cout << endl;
	    */
	    word = readWordIndex (fi);     //read next word
            computeNet (last_word, word);  //compute probability distribution
	    
            //end of file: test on validation data, iterate till convergence
            if (feof(fi)) 
	      break;        

            if (word != -1) 
	      logp += log10 (neu2[vocab[word].class_index + vocab_size].ac * neu2[word].ac);
    	    
    	    if ((logp != logp) || (isinf(logp))) {
    	        printf("\nNumerical error %d %f %f\n", word, neu2[word].ac, 
		       neu2[vocab[word].class_index+vocab_size].ac);
    	        exit(1);
    	    }
	    
            //
            if (bptt > 0) {		//shift memory needed for bptt to next time step
	      for (a = bptt + bptt_block - 1; a > 0; a--) 
		bptt_history[a] = bptt_history[a - 1];
	      bptt_history[0] = last_word;
		
	      for (a = bptt + bptt_block - 1; a > 0; a--) 
		for (b = 0; b < layer1_size; b++) {
		  bptt_hidden[a * layer1_size + b].ac = bptt_hidden[(a - 1) * layer1_size + b].ac;
		  bptt_hidden[a * layer1_size + b].er = bptt_hidden[(a - 1) * layer1_size + b].er;
		}
            }
            //
            learnNet (last_word, word);
            
            copyHiddenLayerToInput ();

            if (last_word != -1) 
	      neu0 [last_word].ac = 0;  //delete previous activation

            last_word = word;
            
            for (a = MAX_NGRAM_ORDER - 1; a > 0; a--) 
	      history [a] = history [a - 1];
            history [0] = last_word;

	    if (independent && (word == 0)) 
	      netReset ();
        }
        fclose(fi);

	now=clock();
    	//printf("%cIter: %3d\tAlpha: %f\t   TRAIN entropy: %.4f    Words/sec: %.1f   ", 13, iter, alpha, -logp/log10(2)/counter, counter/((double)(now-start)/1000000.0));
   
    	if (one_iter==1) {	//no validation data are needed and network is always saved with modified weights
    	    printf("\n");
	    logp=0;
    	    saveNet();
            break;
    	}

        //VALIDATION PHASE
        netFlush();

        fi=fopen(valid_file, "rb");
	if (fi==NULL) {
	    printf("Valid file not found\n");
	    exit(1);
	}
        
        flog=fopen(log_name, "ab");
	if (flog==NULL) {
	    printf("Cannot open log file\n");
	    exit(1);
	}
        
        //fprintf(flog, "Index   P(NET)          Word\n");
        //fprintf(flog, "----------------------------------\n");
        
        last_word=0;
        logp=0;
        wordcn=0;
        while (1) {
            word=readWordIndex(fi);     //read next word
            computeNet(last_word, word);      //compute probability distribution
            if (feof(fi)) break;        //end of file: report LOGP, PPL
            
    	    if (word!=-1) {
    		logp+=log10(neu2[vocab[word].class_index+vocab_size].ac * neu2[word].ac);
        	wordcn++;
    	    }

            /*if (word!=-1)
                fprintf(flog, "%d\t%f\t%s\n", word, neu2[word].ac, vocab[word].word);
            else
                fprintf(flog, "-1\t0\t\tOOV\n");*/

            //learnNet(last_word, word);    //*** this will be in implemented for dynamic models
            copyHiddenLayerToInput ();

            if (last_word != -1) 
	      neu0 [last_word].ac = 0;  //delete previous activation

            last_word=word;
            
            for (a = MAX_NGRAM_ORDER - 1; a > 0; a--) 
	      history [a] = history [a - 1];
            history [0] = last_word;

	    if (independent && (word == 0)) 
	      netReset();
        }
        fclose(fi);
        
        fprintf(flog, "\niter: %d\n", iter);
        fprintf(flog, "valid log probability: %f\n", logp);
        fprintf(flog, "PPL net: %f\n", exp10(-logp/(real)wordcn));
        
        fclose(flog);
    
        printf("VALID entropy: %.4f\n", -logp/log10(2)/wordcn);
        
        counter=0;
	train_cur_pos=0;

        if (logp<llogp)
            restoreWeights();
        else
            saveWeights();

        if (logp*min_improvement<llogp) {
            if (alpha_divide==0) alpha_divide=1;
            else {
                saveNet();
                break;
            }
        }

        if (alpha_divide) alpha/=2;

        llogp=logp;
        logp=0;
        iter++;
        saveNet();
    }
}

void CRnnLM::testNet()
{
    int a, b, word, last_word, wordcn;
    FILE *fi, *flog, *lmprob=NULL;
    real prob_other, log_other, log_combine;
    double d;
    
    restoreNet ();
    
    if (use_lmprob) {
	lmprob=fopen(lmprob_file, "rb");
    }

    //TEST PHASE
    //netFlush();

    fi=fopen(test_file, "rb");
    //sprintf(str, "%s.%s.output.txt", rnnlm_file, test_file);
    //flog=fopen(str, "wb");
    flog=stdout;

    if (debug_mode>1)	{
	if (use_lmprob) {
    	    fprintf(flog, "Index   P(NET)          P(LM)           Word\n");
    	    fprintf(flog, "--------------------------------------------------\n");
	} else {
    	    fprintf(flog, "Index   P(NET)          Word\n");
    	    fprintf(flog, "----------------------------------\n");
	}
    }

    last_word=0;					//last word = end of sentence
    logp=0;
    log_other=0;
    log_combine=0;
    prob_other=0;
    wordcn=0;
    copyHiddenLayerToInput();
    
    if (bptt>0) for (a=0; a<bptt+bptt_block; a++) bptt_history[a]=0;
    for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;
    if (independent) netReset();
    
    while (1) {
        
        word=readWordIndex(fi);		//read next word
        computeNet(last_word, word);		//compute probability distribution
        if (feof(fi)) break;		//end of file: report LOGP, PPL
        
        if (use_lmprob) {
            fscanf(lmprob, "%lf", &d);
    	    prob_other=d;

            goToDelimiter('\n', lmprob);
        }

        if ((word!=-1) || (prob_other>0)) {
    	    if (word==-1) {
    		logp+=-8;		
                //some ad hoc penalty - when mixing different vocabularies, single model score is not real PPL
        	log_combine+=log10(0 * lambda + prob_other*(1-lambda));
    	    } else {
	      // Final softmax prob is computed by multiplying P (class) * P (word)
	      //  
	      logp += log10 (neu2 [vocab [word].class_index + vocab_size].ac * neu2 [word].ac);
	      log_combine += log10 (neu2 [vocab [word].class_index + vocab_size].ac 
				    * neu2 [word].ac 
				    * lambda 
				    + prob_other 
				    * (1 - lambda));
    	    }
    	    log_other+=log10(prob_other);
            wordcn++;
        }

	if (debug_mode>1) {
    	    if (use_lmprob) {
        	if (word!=-1) fprintf(flog, "%d\t%.10f\t%.10f\t%s", word, neu2[vocab[word].class_index+vocab_size].ac *neu2[word].ac, prob_other, vocab[word].word);
        	else fprintf(flog, "-1\t0\t\t0\t\tOOV");
    	    } else {
        	if (word!=-1) fprintf(flog, "%d\t%.10f\t%s", word, neu2[vocab[word].class_index+vocab_size].ac *neu2[word].ac, vocab[word].word);
        	else fprintf(flog, "-1\t0\t\tOOV");
    	    }
    	    
    	    fprintf(flog, "\n");
    	}

        if (dynamic>0) {
            if (bptt>0) {
                for (a=bptt+bptt_block-1; a>0; a--) bptt_history[a]=bptt_history[a-1];
                bptt_history[0]=last_word;
                                    
                for (a=bptt+bptt_block-1; a>0; a--) for (b=0; b<layer1_size; b++) {
                    bptt_hidden[a*layer1_size+b].ac=bptt_hidden[(a-1)*layer1_size+b].ac;
                    bptt_hidden[a*layer1_size+b].er=bptt_hidden[(a-1)*layer1_size+b].er;
        	}
            }
            //
            alpha=dynamic;
    	    learnNet(last_word, word);    //dynamic update
    	}
        copyHiddenLayerToInput();
        
        if (last_word!=-1) neu0[last_word].ac=0;  //delete previous activation

        last_word=word;
        
        for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
        history[0]=last_word;

	if (independent && (word==0)) netReset();
    }
    fclose(fi);
    if (use_lmprob) fclose(lmprob);

    //write to log file
    if (debug_mode>0) {
	fprintf(flog, "\ntest log probability: %f\n", logp);
	if (use_lmprob) {
    	    fprintf(flog, "test log probability given by other lm: %f\n", log_other);
    	    fprintf(flog, "test log probability %f*rnn + %f*other_lm: %f\n", lambda, 1-lambda, log_combine);
	}

	fprintf(flog, "\nPPL net: %f\n", exp10(-logp/(real)wordcn));
	if (use_lmprob) {
    	    fprintf(flog, "PPL other: %f\n", exp10(-log_other/(real)wordcn));
    	    fprintf(flog, "PPL combine: %f\n", exp10(-log_combine/(real)wordcn));
	}
    }
    
    fclose(flog);
}

void CRnnLM::testNbest()
{
    int a, word, last_word, wordcn;
    FILE *fi, *flog, *lmprob=NULL;
    float prob_other; //has to be float so that %f works in fscanf
    real log_other, log_combine, senp;
    //int nbest=-1;
    int nbest_cn=0;
    char ut1[MAX_STRING], ut2[MAX_STRING];

    restoreNet();
    computeNet(0, 0);
    copyHiddenLayerToInput();
    saveContext();
    saveContext2();
    
    if (use_lmprob) {
	lmprob=fopen(lmprob_file, "rb");
    } else lambda=1;		//!!! for simpler implementation later

    //TEST PHASE
    //netFlush();
    
    for (a=0; a<MAX_NGRAM_ORDER; a++) history[a]=0;

    if (!strcmp(test_file, "-")) fi=stdin; else fi=fopen(test_file, "rb");
    
    //sprintf(str, "%s.%s.output.txt", rnnlm_file, test_file);
    //flog=fopen(str, "wb");
    flog=stdout;

    last_word=0;		//last word = end of sentence
    logp=0;
    log_other=0;
    prob_other=0;
    log_combine=0;
    wordcn=0;
    senp=0;
    strcpy(ut1, (char *)"");
    while (1) {
	if (last_word==0) {
	    fscanf(fi, "%s", ut2);
	    //save context after processing first sentence in nbest
	    if (nbest_cn==1) 
	      saveContext2();	
	    
	    if (strcmp(ut1, ut2)) {
		strcpy(ut1, ut2);
		nbest_cn=0;
		restoreContext2();
		saveContext();
	    } else restoreContext();
	    
	    nbest_cn++;
	    
	    copyHiddenLayerToInput();
        }
    
	
	word=readWordIndex(fi);     //read next word
	if (lambda>0) computeNet(last_word, word);      //compute probability distribution
        if (feof(fi)) break;        //end of file: report LOGP, PPL
        
        
        if (use_lmprob) {
            fscanf(lmprob, "%f", &prob_other);
            goToDelimiter('\n', lmprob);
        }
        
        if (word!=-1)
        neu2[word].ac*=neu2[vocab[word].class_index+vocab_size].ac;
        
        if (word!=-1) {
            logp+=log10(neu2[word].ac);
    	    
            log_other+=log10(prob_other);
            
            log_combine+=log10(neu2[word].ac*lambda + prob_other*(1-lambda));
            
            senp+=log10(neu2[word].ac*lambda + prob_other*(1-lambda));
            
            wordcn++;
        } else {
    	    //assign to OOVs some score to correctly rescore nbest lists, reasonable value can be less than 1/|V| or backoff LM score (in case it is trained on more data)
    	    //this means that PPL results from nbest list rescoring are not true probabilities anymore (as in open vocabulary LMs)
    	    
    	    real oov_penalty=-5;	//log penalty
    	    
    	    if (prob_other!=0) {
    		logp+=log10(prob_other);
    		log_other+=log10(prob_other);
    		log_combine+=log10(prob_other);
    		senp+=log10(prob_other);
    	    } else {
    		logp+=oov_penalty;
    		log_other+=oov_penalty;
    		log_combine+=oov_penalty;
    		senp+=oov_penalty;
    	    }
    	    wordcn++;
        }
        
        //learnNet(last_word, word);    //*** this will be in implemented for dynamic models
        copyHiddenLayerToInput();

        if (last_word!=-1) neu0[last_word].ac=0;  //delete previous activation
        
        if (word==0) {		//write last sentence log probability / likelihood
    	    fprintf(flog, "%f\n", senp);
    	    senp=0;
	}

        last_word=word;
        
        for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
        history[0]=last_word;

	if (independent && (word==0)) netReset();
    }
    fclose(fi);
    if (use_lmprob) fclose(lmprob);

    if (debug_mode>0) {
	printf("\ntest log probability: %f\n", logp);
	if (use_lmprob) {
    	    printf("test log probability given by other lm: %f\n", log_other);
    	    printf("test log probability %f*rnn + %f*other_lm: %f\n", lambda, 1-lambda, log_combine);
	}

	printf("\nPPL net: %f\n", exp10(-logp/(real)wordcn));
	if (use_lmprob) {
    	    printf("PPL other: %f\n", exp10(-log_other/(real)wordcn));
    	    printf("PPL combine: %f\n", exp10(-log_combine/(real)wordcn));
	}
    }

    fclose(flog);
}

void CRnnLM::testGen()
{
    int i, word, cla, last_word, wordcn, c, b, a=0;
    real f, g, sum, val;
    
    restoreNet();
    
    word=0;
    last_word=0;					//last word = end of sentence
    wordcn=0;
    copyHiddenLayerToInput();
    while (wordcn<gen) {
        computeNet(last_word, 0);		//compute probability distribution
        
        f=random(0, 1);
        g=0;
        i=vocab_size;
        while ((g<f) && (i<layer2_size)) {
    	    g+=neu2[i].ac;
    	    i++;
        }
        cla=i-1-vocab_size;
        
        if (cla>class_size-1) cla=class_size-1;
        if (cla<0) cla=0;
        
        //
        // !!!!!!!!  THIS WILL WORK ONLY IF CLASSES ARE CONTINUALLY DEFINED IN VOCAB !!! (like class 10 = words 11 12 13; not 11 12 16)  !!!!!!!!
        // forward pass 1->2 for words
        for (c=0; c<class_cn[cla]; c++) neu2[class_words[cla][c]].ac=0;
        matrixXvector(neu2, neu1, syn1, layer1_size, class_words[cla][0], class_words[cla][0]+class_cn[cla], 0, layer1_size, 0);
        
        //apply direct connections to words
	if (word!=-1) if (direct_size>0) {
    	    unsigned long long hash[MAX_NGRAM_ORDER];

            for (a=0; a<direct_order; a++) hash[a]=0;

            for (a=0; a<direct_order; a++) {
                b=0;
                if (a>0) if (history[a-1]==-1) break;
                hash[a]=PRIMES[0]*PRIMES[1]*(unsigned long long)(cla+1);

                for (b=1; b<=a; b++) hash[a]+=PRIMES[(a*PRIMES[b]+b)%PRIMES_SIZE]*(unsigned long long)(history[b-1]+1);
                hash[a]=(hash[a]%(direct_size/2))+(direct_size)/2;
    	    }

    	    for (c=0; c<class_cn[cla]; c++) {
        	a=class_words[cla][c];

        	for (b=0; b<direct_order; b++) if (hash[b]) {
    		    neu2[a].ac+=syn_d[hash[b]];
            	    hash[b]++;
        	    hash[b]=hash[b]%direct_size;
    	        } else break;
    	    }
	}
        
        //activation 2   --softmax on words
	sum=0;
    	for (c=0; c<class_cn[cla]; c++) {
    	    a=class_words[cla][c];
    	    if (neu2[a].ac>50) neu2[a].ac=50;  //for numerical stability
    	    if (neu2[a].ac<-50) neu2[a].ac=-50;  //for numerical stability
    	    val=FAST_EXP(neu2[a].ac);
    	    sum+=val;
    	    neu2[a].ac=val;
    	}
    	for (c=0; c<class_cn[cla]; c++) neu2[class_words[cla][c]].ac/=sum;
	//
	
	f=random(0, 1);
        g=0;
        /*i=0;
        while ((g<f) && (i<vocab_size)) {
    	    g+=neu2[i].ac;
    	    i++;
        }*/
        for (c=0; c<class_cn[cla]; c++) {
    	    a=class_words[cla][c];
    	    g+=neu2[a].ac;
    	    if (g>f) break;
        }
        word=a;
        
	if (word>vocab_size-1) word=vocab_size-1;
        if (word<0) word=0;

	//printf("%s %d %d\n", vocab[word].word, cla, word);
	if (word!=0)
	    printf("%s ", vocab[word].word);
	else
	    printf("\n");

        copyHiddenLayerToInput();

        if (last_word!=-1) neu0[last_word].ac=0;  //delete previous activation

        last_word=word;
        
        for (a=MAX_NGRAM_ORDER-1; a>0; a--) history[a]=history[a-1];
        history[0]=last_word;

	if (independent && (word==0)) netReset();
        
        wordcn++;
    }
}
