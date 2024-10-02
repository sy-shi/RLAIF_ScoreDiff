# TS-LLM
First build the conda environment with environment.yml. Then
```
conda activate rllib_2.2
cd TS-LLM
```

## I. Collect LLM Ranking Results as Training Data of Reward Functions
Note: Please set `sample_size` as 6000 when running the following command to collect data from LLM model.
```
python generate_ranking_data.py --use_llm True --sample_size 6000 --filename xxx.pkl
```
You can specify the LLM model in `value_est()` in **apis.py**.

The collected data is stored in the `filename` passed in. The pkl file stores the data as a dictionary , which has three fields, `input`, `target` and `llm_target`. Each field corresponds to a list with length of the `sample_size` passed in. Each item in `input` is also a list, containing two observation of a state pair as well as the action between these two states. `target` is the ground truth ranking. `llm-target` is the rank given by LLM.


## Integrate dataset
Integrate the data from original llama data and the newly collected data for CC.
The new data set's name is set in this script as `ranking_data_train_env3-llama_family_cced.pkl`
```
cd TS-LLM
python integrate_data.py
```

## process data with CC
modify the `training_data_file` name where you want to store the processed data at line 338 in consistent_check.py
```
python consistent_check.py --mode <consistency check mode, ccc for correct, cccr for correct and remove> --threshold <0.6 or 0.8>
```
for example, 
```
python consistent_check.py --mode ccc --threshold 0.6
```
This will do consistency check based on correct, with threshold 0.6. And the correspondingly generated file can be named as `ranking_data_train_env3-llama_ccc2_thres06.pkl` (2 representing this is our second time doing cc).

Also, remember to record the dataset size after CC.

For each mode and parameter, train reward model
In train_rm.py, modify the file name of `GT_FILE` in line 287 and 290, set `VALI_DATA_SIZE=500` for training RM. Modify `dataset_size` at line 323 to that of your current dataset.
```
python train_rm.py --mode train
```

The result is stored in ./LLM/RM/models/reward_model.pth

To test, set `VALI_DATA_SIZE=30000`
```
python train_rm.py --mode validate --model_path ./LLM/RM/models/reward_model.pth
```

**REMEMBER TO MODIFY THE NAME OF REWARD MODEL BEFORE TRAINING RM WITH OTHER DATA SETS!!!!!!**

The name of RMs should be
1. reward_model_llama_ccc2_thres06.pth
2. reward_model_llama_ccc2_thres08.pth
3. reward_model_llama_cccr2_thres06.pth
4. reward_model_llama_cccr2_thres08.pth

## Train RL agents
To train agent after getting all required RMs, run
```
bash env3_exp1_cc_part2.sh
```