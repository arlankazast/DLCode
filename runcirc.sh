#!/bin/bash

# using a free gpu
# sudo userdocker run -it nvcr.io/nvidia/tensorflow:20.02-tf2-py3 /home/nabeel/API/runcirc.sh

#working dir
cd /Users/arlan/Downloads/DL_code

#for models in "Wang_Hybrid" "Zhou_Hybrid"
for models in "Wang_Hybrid"
do
echo "Signals classification"
"/arlan/Anaconda3/envs/DL_code/python" main.py "$models"
done

echo "Finished processing"