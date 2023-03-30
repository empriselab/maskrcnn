#!/bin/bash

# Author: Thomas Patton (tjp93)
# Program: create_train_data.sh
#
# This .sh script calls the `annotate.py` program in a for loop with each index `i` representing a bagfile number

for i in {2..45}
do
    python annotate.py $i -v 1 --display False
done