##############################################
#CLONING THE REPOSITORY AND HETTING THE DATA
##############################################

import os

# cloning the directory to work in

!git clone https://github.com/1eg1on/nlp-lm
os.chdir(os.getcwd())


# installing the requirements

!pip install -r requirements.txt

# creating the data-folder and downoading the main body of the dataset

!mkdir data
!mkdir data/fever-data

# We use the data used in the baseline paper
!wget -O data/fever-data/train.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/train.jsonl
!wget -O data/fever-data/dev.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/paper_dev.jsonl
!wget -O data/fever-data/test.jsonl https://s3-eu-west-1.amazonaws.com/fever.public/paper_test.jsonl


##############################################
#OBTAIN WIKI
##############################################

!wget https://s3-eu-west-1.amazonaws.com/fever.public/wiki-pages.zip
!unzip wiki-pages.zip -d data




##############################################
#DATABASE CREATION
##############################################


# we use the pre-written script to create a database we shortly will be working with

# Need to pre-install stopwords module in case in is not done yet, used in the script

import nltk
nltk.download('stopwords')

!python build_db.py data/wiki-pages data/single --num-files 1
!python build_db.py data/wiki-pages data/fever --num-files 5



##############################################
#WORKING WITH MATRIXES
##############################################



# Building the matrixes to work with further
nltk.download('punkt')

!python build_count_matrix.py data/fever data/index


!python merge_count_matrix.py data/index data/index

# testing the TF-IDF approach further, reconstructing the matrix using the pre-written script

!python reweight_count_matrix.py data/index/count-ngram\=1-hash\=16777216.npz data/index --model tfidf



##############################################

# additional sampling for NEI. TODO: reimplement oracle.read()

##############################################

def sampling_for_NEI(oracle, num_docs=5, num_sents=5):
    names = ['training','dev','test']
    for name in names:
        print('Working on {} split'.format(name))
        original_path = 'data/fever-data/{}.jsonl'.format(name)
        sampling_path = 'data/fever-data/{}_sampled.jsonl'.format(name)
        with open(original_path, "r") as f:
            with open(sampling_path, "w+") as f2:
                for line in tqdm(f.readlines()):
                    line = json.loads(line)

                    if name == 'dev' or name == 'test' or line["label"] == "NOT ENOUGH INFO":
                        evidences = oracle.read(line['claim'], num_docs=num_docs, num_sents=num_sents).keys()
                        line['evidence'] = [[[0,0,ev[0],ev[1]] for ev in evidences]]

                    f2.write(json.dumps(line) + "\n")
                    
                    
sampling_for_NEI(oracle)



################################## from here TODO is stated


'''

MAX


0. Check if we can reuse TFIDF text- extraction separately from the generaly (mb we can reimplement it) search fever.py for the answer

1. Check out what are the outputs of the previous part

5. Assign evidences for closest sentences as evidences for claim

6. Give out vectors based either on the evidences themselves or on the distribution of closeness measures

7 Fit the model*


----------


GABRIEL

2. Check out what do we need as an input dor GenSim 

3. Implement Doc2Vec with cut sentences (cut out of texts) How many features---50? 30? 20?


-----------

4. Find closest sentences using vectorization

5. Assign evidences for closest sentences as evidences for claim

6. Give out vectors based either on the evidences themselves or on the distribution of closeness measures

7 Fit the model*


'''