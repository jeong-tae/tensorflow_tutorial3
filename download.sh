#!/bin/bash

DATA_PATH="./data"

#################################
# Stanford's SNLI
# # # http://nlp.stanford.edu/pubs/snli_paper.pdf
# # # http://arxiv.org/abs/1601.06733
# #################################

SNLI_PATH="$DATA_PATH/snli"
SNLI_FNAME="snli_1.0"
SNLI_URL="http://nlp.stanford.edu/projects/snli/snli_1.0.zip"

if [ ! -d $SNLI_PATH ]
then
    echo " [*] Download SNLI dataset..."
    mkdir -p $SNLI_PATH
    cd $SNLI_PATH && { curl -O $SNLI_URL; cd -; }
    unzip "$SNLI_PATH/$SNLI_FNAME.zip" -d "$SNLI_PATH"
    rm -rf "$SNLI_PATH/__MACOSX"
else
    echo " [*] SNLI already exists"
fi

######################################
# Cornell movie dialogs corpus
# # http://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
######################################


DIALOG_PATH="$DATA_PATH/dialog"
DIALOG_FNAME="cornell_movie_dialogs_corpus"
DIALOG_URL="http://www.mpi-sws.org/~cristian/data/cornell_movie_dialogs_corpus.zip"

if [ ! -d $DIALOG_PATH ]
then
    echo " [*] Download cornell movie dialog dataset..."
    mkdir -p $DIALOG_PATH
    cd $DIALOG_PATH && { curl -O $DIALOG_URL; cd -; }
    unzip "$DIALOG_PATH/$DIALOG_FNAME.zip" -d "$DIALOG_PATH"
    rm -rf "$DIALOG_PATH/__MACOSX"
else
    echo " [*] Dialog already exists"
fi
