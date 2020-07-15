#!/usr/bin/env bash
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#
# Revisions:
#  * July 2020: Thamme Gowda; simplification and generalization for any two pairs of languages
#
SCRIPTS="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
set -e


#
# Data preprocessing configuration
#
N_MONO=5000000  # number of monolingual sentences for each language
CODES=32000     # number of BPE codes
 # number of threads in data preprocessing
N_THREADS=$(python -c "import multiprocessing as mp; print(min(16, mp.cpu_count()))")
#LWLL_DATA=/home/tgowda/work/lwll/runs/y1-dev/data
TOOLS_PATH=$PWD/tools

#
# Read arguments
#
DATA_PATH=
MONO_PREF=
PARA_VAL_PREF=
POSITIONAL=()
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --src)
      SRC="$2"; shift 2;;
    --tgt)
      TGT="$2"; shift 2;;
    --mono)
      MONO_PREF="$2"; shift 2;;
    --para_val)
      PARA_VAL_PREF="$2"; shift 2;;
    --data)
      DATA_PATH="$2"; shift 2;;
    --reload_codes)
      RELOAD_CODES="$2"; shift 2;;
    --reload_vocab)
      RELOAD_VOCAB="$2"; shift 2;;
    *)
    POSITIONAL+=("$1")
    shift;;
  esac
done
set -- "${POSITIONAL[@]}"

quit(){  # quit with error message
  printf "ERROR: $2\n"
  exit $1
}
#
# Check parameters
#
[[ -n "$SRC" ]] || quit 1 "--src not provided"
[[ -n "$TGT" ]] || quit 1 "--tgt not provided"
[[ -n "$DATA_PATH" ]] || quit 1 "--data path/to/store/data is required"
[[ -n "$MONO_PREF" ]] ||quit 1 "--mono mono text path prefix required"
[[ -n "$PARA_VAL_PREF" ]] || quit 1 "--para_val validation data (bitext) required"

[[ "$SRC" != "$TGT" ]] || quit 1  "source and target cannot be identical"
[[ "$SRC" < "$TGT" ]] || quit 1  "please ensure SRC < TGT"
[[ -z "$RELOAD_CODES"  && ! -f "$RELOAD_CODES" ]] || quit 1 "cannot locate BPE codes"
[[ -z "$RELOAD_VOCAB"  && ! -f "$RELOAD_VOCAB" ]] || quit 1  "cannot locate vocabulary"
if [[ -z "$RELOAD_CODES" &&  -n "$RELOAD_VOCAB" || -n "$RELOAD_CODES" && -z "$RELOAD_VOCAB" ]]
then
  echo "BPE codes should be provided if and only if vocabulary is also provided";
  exit
fi

SRC_ORIG=$MONO_PREF.$SRC
TGT_ORIG=$MONO_PREF.$TGT
PARA_SRC_VALID_ORIG=$PARA_VAL_PREF.$SRC
PARA_TGT_VALID_ORIG=$PARA_VAL_PREF.$TGT
#PARA_SRC_TEST_ORIG=$LWLL_DATA/tests/UNv1_test.ara
#PARA_TGT_TEST_ORIG=$LWLL_DATA/tests/UNv1_test.eng

[[ -f $SRC_ORIG ]] || { echo "SRC_ORIG $SRC_ORIG not found"; exit 2; }
[[ -f $TGT_ORIG ]] || { echo "TGT_ORIG $TGT_ORIG not found"; exit 2; }

for var in {SRC,TGT}_ORIG  PARA_{SRC,TGT}_VALID_ORIG; do
    eval "path=\$$var"
    if [[ ! -f "$path" ]]; then
	    echo "$var $path not found"
	    exit 3;
    fi
done

#
# Initialize tools and data paths
#
# main paths
#MAIN_PATH=$PWD
#TOOLS_PATH=$PWD/tools

MONO_PATH=$DATA_PATH/mono
PARA_PATH=$DATA_PATH/para
PROC_PATH=$DATA_PATH/processed/$SRC-$TGT

# create paths
[[ -d $TOOLS_PATH ]] || mkdir -p $TOOLS_PATH
# install tools TODO: pass argument
$SCRIPTS/install-tools.sh $TOOLS_PATH || {
  echo "Tools setup failed. Exiting..."
  exit 2
}

mkdir -p $DATA_PATH
mkdir -p $MONO_PATH
mkdir -p $PARA_PATH
mkdir -p $PROC_PATH

PREPROCESS="python -m unmass.preprocess"
# moses
MOSES=$TOOLS_PATH/mosesdecoder
REPLACE_UNICODE_PUNCT=$MOSES/scripts/tokenizer/replace-unicode-punctuation.perl
NORM_PUNC=$MOSES/scripts/tokenizer/normalize-punctuation.perl
REM_NON_PRINT_CHAR=$MOSES/scripts/tokenizer/remove-non-printing-char.perl
TOKENIZER=$MOSES/scripts/tokenizer/tokenizer.perl
#INPUT_FROM_SGM=$MOSES/scripts/ems/support/input-from-sgm.perl

# fastBPE
#FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$TOOLS_PATH/fastBPE/fast

# raw and tokenized files
#SRC_ORIG=$MONO_PATH.$SRC
#TGT_ORIG=$MONO_PATH.$TGT
[[ -f "$SRC_ORIG" ]] || { echo "$SRC_ORIG not found;" exit 2; }
[[ -f "$TGT_ORIG" ]] || { echo "$TGT_ORIG not found;" exit 2; }

SRC_RAW=$MONO_PATH/$SRC/all.$SRC
TGT_RAW=$MONO_PATH/$TGT/all.$TGT
SRC_TOK=$SRC_RAW.tok
TGT_TOK=$TGT_RAW.tok

# BPE / vocab files
BPE_CODES=$PROC_PATH/codes
SRC_VOCAB=$PROC_PATH/vocab.$SRC
TGT_VOCAB=$PROC_PATH/vocab.$TGT
FULL_VOCAB=$PROC_PATH/vocab.$SRC-$TGT

# train / valid / test monolingual BPE data
SRC_TRAIN_BPE=$PROC_PATH/train.$SRC
TGT_TRAIN_BPE=$PROC_PATH/train.$TGT
SRC_VALID_BPE=$PROC_PATH/valid.$SRC
TGT_VALID_BPE=$PROC_PATH/valid.$TGT
#SRC_TEST_BPE=$PROC_PATH/test.$SRC
#TGT_TEST_BPE=$PROC_PATH/test.$TGT

# valid/src
PARA_SRC_VALID=$PARA_PATH/dev/dev.$SRC
PARA_TGT_VALID=$PARA_PATH/dev/dev.$TGT
#PARA_SRC_TEST=$PARA_PATH/test/test.$SRC
#PARA_TGT_TEST=$PARA_PATH/test/test.$TGT

# valid / test parallel BPE data
PARA_SRC_VALID_BPE=$PROC_PATH/valid.$SRC-$TGT.$SRC
PARA_TGT_VALID_BPE=$PROC_PATH/valid.$SRC-$TGT.$TGT
PARA_SRC_TEST_BPE=$PROC_PATH/test.$SRC-$TGT.$SRC
PARA_TGT_TEST_BPE=$PROC_PATH/test.$SRC-$TGT.$TGT


mkdir -p $MONO_PATH/$SRC  $MONO_PATH/$TGT $PARA_PATH/dev $PARA_PATH/test
#cd $MONO_PATH

# concatenate monolingual data files
if ! [[ -f "$SRC_RAW" ]]; then
  echo " $SRC monolingual data..."
  cat $SRC_ORIG | shuf | head -n $N_MONO > $SRC_RAW
fi
if ! [[ -f "$TGT_RAW" ]]; then
  echo " $TGT monolingual data..."
  cat $TGT_ORIG | shuf | head -n $N_MONO > $TGT_RAW
fi
echo "$SRC selected monolingual data in: $SRC_RAW"
echo "$TGT selected monolingual data in: $TGT_RAW"

# # check number of lines
# if ! [[ "$(wc -l < $SRC_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines does not match! Be sure you have $N_MONO sentences in your $SRC monolingual data."; exit; fi
# if ! [[ "$(wc -l < $TGT_RAW)" -eq "$N_MONO" ]]; then echo "ERROR: Number of lines does not match! Be sure you have $N_MONO sentences in your $TGT monolingual data."; exit; fi

# preprocessing commands - special case for Romanian

SRC_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $SRC | $REM_NON_PRINT_CHAR | $TOKENIZER -l $SRC -no-escape -threads $N_THREADS"
TGT_PREPROCESSING="$REPLACE_UNICODE_PUNCT | $NORM_PUNC -l $TGT | $REM_NON_PRINT_CHAR |  $TOKENIZER -l $TGT -no-escape -threads $N_THREADS"


# tokenize data
if ! [[ -f "$SRC_TOK" ]]; then
  echo "Tokenize $SRC monolingual data..."
  eval "cat $SRC_RAW | $SRC_PREPROCESSING > $SRC_TOK"
fi

if ! [[ -f "$TGT_TOK" ]]; then
  echo "Tokenize $TGT monolingual data..."
  eval "cat $TGT_RAW | $TGT_PREPROCESSING > $TGT_TOK"
fi
echo "$SRC monolingual data tokenized in: $SRC_TOK"
echo "$TGT monolingual data tokenized in: $TGT_TOK"

# reload BPE codes
#cd $MAIN_PATH
if [ ! -f "$BPE_CODES" ] && [ -f "$RELOAD_CODES" ]; then
  echo "Reloading BPE codes from $RELOAD_CODES ..."
  cp $RELOAD_CODES $BPE_CODES
fi

# learn BPE codes
if [ ! -f "$BPE_CODES" ]; then
  echo "Learning BPE codes..."
  $FASTBPE learnbpe $CODES $SRC_TOK $TGT_TOK > $BPE_CODES
fi
echo "BPE learned in $BPE_CODES"

# apply BPE codes
if ! [[ -f "$SRC_TRAIN_BPE" ]]; then
  echo "Applying $SRC BPE codes..."
  $FASTBPE applybpe $SRC_TRAIN_BPE $SRC_TOK $BPE_CODES
fi
if ! [[ -f "$TGT_TRAIN_BPE" ]]; then
  echo "Applying $TGT BPE codes..."
  $FASTBPE applybpe $TGT_TRAIN_BPE $TGT_TOK $BPE_CODES
fi
echo "BPE codes applied to $SRC in: $SRC_TRAIN_BPE"
echo "BPE codes applied to $TGT in: $TGT_TRAIN_BPE"

# extract source and target vocabulary
if ! [[ -f "$SRC_VOCAB" && -f "$TGT_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TRAIN_BPE > $SRC_VOCAB
  $FASTBPE getvocab $TGT_TRAIN_BPE > $TGT_VOCAB
fi
echo "$SRC vocab in: $SRC_VOCAB"
echo "$TGT vocab in: $TGT_VOCAB"

# reload full vocabulary
#cd $MAIN_PATH
if [ ! -f "$FULL_VOCAB" ] && [ -f "$RELOAD_VOCAB" ]; then
  echo "Reloading vocabulary from $RELOAD_VOCAB ..."
  cp $RELOAD_VOCAB $FULL_VOCAB
fi

# extract full vocabulary
if ! [[ -f "$FULL_VOCAB" ]]; then
  echo "Extracting vocabulary..."
  $FASTBPE getvocab $SRC_TRAIN_BPE $TGT_TRAIN_BPE > $FULL_VOCAB
fi
echo "Full vocab in: $FULL_VOCAB"

# binarize data
if ! [[ -f "$SRC_TRAIN_BPE.pth" ]]; then
  echo "Binarizing $SRC data..."
  $PREPROCESS $FULL_VOCAB $SRC_TRAIN_BPE
fi
if ! [[ -f "$TGT_TRAIN_BPE.pth" ]]; then
  echo "Binarizing $TGT data..."
  $PREPROCESS $FULL_VOCAB $TGT_TRAIN_BPE
fi
echo "$SRC binarized data in: $SRC_TRAIN_BPE.pth"
echo "$TGT binarized data in: $TGT_TRAIN_BPE.pth"


#
# Download parallel data (for evaluation only)
#
#cd $PARA_PATH

echo "Tokenizing validation data..."
eval "cat $PARA_SRC_VALID_ORIG | $SRC_PREPROCESSING > $PARA_SRC_VALID"
eval "cat $PARA_TGT_VALID_ORIG | $TGT_PREPROCESSING > $PARA_TGT_VALID"
#eval "cat $PARA_SRC_TEST_ORIG | $SRC_PREPROCESSING > $PARA_SRC_TEST"
#eval "cat $PARA_TGT_TEST_ORIG | $TGT_PREPROCESSING > $PARA_TGT_TEST"

echo "Applying BPE to valid and test files..."
$FASTBPE applybpe $PARA_SRC_VALID_BPE $PARA_SRC_VALID $BPE_CODES $SRC_VOCAB
$FASTBPE applybpe $PARA_TGT_VALID_BPE $PARA_TGT_VALID $BPE_CODES $TGT_VOCAB
#$FASTBPE applybpe $PARA_SRC_TEST_BPE  $PARA_SRC_TEST  $BPE_CODES $SRC_VOCAB
#$FASTBPE applybpe $PARA_TGT_TEST_BPE  $PARA_TGT_TEST  $BPE_CODES $TGT_VOCAB

echo "Binarizing data..."
rm -f $PARA_SRC_VALID_BPE.pth $PARA_TGT_VALID_BPE.pth $PARA_SRC_TEST_BPE.pth $PARA_TGT_TEST_BPE.pth
$PREPROCESS $FULL_VOCAB $PARA_SRC_VALID_BPE
$PREPROCESS $FULL_VOCAB $PARA_TGT_VALID_BPE
#$PREPROCESS $FULL_VOCAB $PARA_SRC_TEST_BPE
#$PREPROCESS $FULL_VOCAB $PARA_TGT_TEST_BPE


#
# Link monolingual validation and test data to parallel data
#
ln -sf "$(basename $PARA_SRC_VALID_BPE.pth)" $SRC_VALID_BPE.pth
ln -sf "$(basename $PARA_TGT_VALID_BPE.pth)" $TGT_VALID_BPE.pth
#ln -sf $PARA_SRC_TEST_BPE.pth  $SRC_TEST_BPE.pth
#ln -sf $PARA_TGT_TEST_BPE.pth  $TGT_TEST_BPE.pth


#
# Summary
#
echo ""
echo "===== Data summary"
echo "Monolingual training data:"
echo "    $SRC: $SRC_TRAIN_BPE.pth"
echo "    $TGT: $TGT_TRAIN_BPE.pth"
echo "Monolingual validation data:"
echo "    $SRC: $SRC_VALID_BPE.pth"
echo "    $TGT: $TGT_VALID_BPE.pth"
#echo "Monolingual test data:"
#echo "    $SRC: $SRC_TEST_BPE.pth"
#echo "    $TGT: $TGT_TEST_BPE.pth"
echo "Parallel validation data:"
echo "    $SRC: $PARA_SRC_VALID_BPE.pth"
echo "    $TGT: $PARA_TGT_VALID_BPE.pth"
#echo "Parallel test data:"
#echo "    $SRC: $PARA_SRC_TEST_BPE.pth"
#echo "    $TGT: $PARA_TGT_TEST_BPE.pth"
echo ""