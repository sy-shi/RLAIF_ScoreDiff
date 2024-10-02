#!/bin/bash

# Path to the YAML file
FILE="/home/ssy/Desktop/teacher-student_code/TS_LLM/config/doubledoor_.yaml"
BASE_COMMAND="python train.py --config=doubledoor_ --mode=train --max-steps=300000"

# Check if the YAML file exists
if [ ! -f "$FILE" ]; then
    echo "Error: $FILE does not exist."
    exit 1
fi

SRC=$(pwd)
TEACHER_MODEL_PATHS=("$SRC/LLM/RM/models/reward_model_llama6000.pth" "$SRC/LLM/RM/models/reward_model6k_new.pth" "$SRC/LLM/RM/models/reward_model_gpt6000.pth" "$SRC/LLM/RM/models/reward_model_mixtral6000.pth")

for TEACHER_MODEL_PATH in "${TEACHER_MODEL_PATHS[@]}"; do
    echo "$TEACHER_MODEL_PATH"
done

for i in {1..5}; do
    for TEACHER_MODEL_PATH in "${TEACHER_MODEL_PATHS[@]}"; do
        # Update the teacher_model_path in the YAML file
        yq e ".BASE_CONFIG.teacher_model_path = \"$TEACHER_MODEL_PATH\"" -i "$FILE"
        echo "YAML file updated with teacher_model_path: $TEACHER_MODEL_PATH."
        if [[ "$TEACHER_MODEL_PATH" == *"llama6000"* ]]; then
            experiment_name="llama"
        elif [[ "$TEACHER_MODEL_PATH" == *"gpt6000"* ]]; then
            experiment_name="gpt"
        elif [[ "$TEACHER_MODEL_PATH" == *"6k_new"* ]]; then
            experiment_name="gt"
        fi
        $BASE_COMMAND --name="env3_${experiment_name}_trial$i"
    done
done