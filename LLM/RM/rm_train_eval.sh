#!/bin/bash

# Shell script to uncomment a line in a file

# Check for the correct number of arguments
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <line_number>"
    exit 1
fi

line_number=$1
file_name=/home/ssy/Desktop/teacher-student_code/TS_LLM/LLM/RM/train_rm.py

# Use sed to remove a '#' from the beginning of the specified line
# -i option edits the file in place
# "s/^#//" removes the first '#' at the start of the line
# "${line_number}" specifies the line number to edit
sed -i "${line_number}s/^#//" "$file_name"

echo "Line $line_number in $file_name has been uncommented."


python3 /home/ssy/Desktop/teacher-student_code/TS_LLM/LLM/RM/train_rm.py --mode train
#/bin/python3 /home/sunnylin/transfer_learning_ts/LLM/RM/train_rm.py --agent 1


sed -i "${line_number}s/^/#/" "$file_name"

echo "Line $line_number in $file_name has been commented out."

#source /path/to/anaconda3/etc/profile.d/conda.sh
# source activate rllib_2.2
# python train.py --mode=test_rm
# source deactivate