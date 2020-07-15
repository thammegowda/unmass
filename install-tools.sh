#!/usr/bin/env bash
# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# NOTICE FILE in the root directory of this source tree.
#

set -e

#lg=$1  # input language

TOOLS_PATH=${1:-${PWD}/tools}   # get $1 or default PWD/tools

# tools
MOSES_DIR=$TOOLS_PATH/mosesdecoder
FASTBPE_DIR=$TOOLS_PATH/fastBPE
FASTBPE=$FASTBPE_DIR/fast
WMT16_SCRIPTS=$TOOLS_PATH/wmt16-scripts
PREPROCESS="python -m unmass.preprocess"

# tools path
mkdir -p $TOOLS_PATH

python -m unmass.preprocess -h &> /dev/null  || {
  echo "Cant find 'python -m unmass.preprocess'; Please fix your PYTHONPATH or install unmass"
  exit 2
}


#
# Download and install tools
#
cd $TOOLS_PATH

# Download Moses
if [ ! -d "$MOSES_DIR" ]; then
  echo "Cloning Moses from GitHub repository..."
  git clone https://github.com/moses-smt/mosesdecoder.git
fi

# Download fastBPE
if [ ! -d "$FASTBPE_DIR" ]; then
  echo "Cloning fastBPE from GitHub repository..."
  git clone https://github.com/glample/fastBPE
fi

# Compile fastBPE
if [ ! -f "$FASTBPE" ]; then
  echo "Compiling fastBPE..."
  cd fastBPE
  g++ -std=c++11 -pthread -O3 fastBPE/main.cc -IfastBPE -o fast
  cd ..
fi

# Download Sennrich's tools
if [ ! -d "$WMT16_SCRIPTS" ]; then
  echo "Cloning WMT16 preprocessing scripts..."
  git clone https://github.com/rsennrich/wmt16-scripts.git
fi

