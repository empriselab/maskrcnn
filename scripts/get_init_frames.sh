#!/bin/bash
# calls python script with numbers representing number of bagfiles
for i in {26..45}
do
    python get_init_frames.py $i
done