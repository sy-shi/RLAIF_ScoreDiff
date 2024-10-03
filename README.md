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

The collected data is stored in the `filename` passed in. The pkl file stores the data as a dictionary , which has three fields, `input`, `target` and `llm_target`. Each field corresponds to a list with length of the `sample_size` passed in. Each item in `input` is also a list, containing the last and the current states as well as the action between these two states. `target` is the ground truth ranking. `llm-target` is the rank given by LLM.

Without the `use_llm` flag here, this command only collects state pairs with GT ranking.

## II. Scoring Model Training
```
python3 train_rm.py --mode=train
```
Trained models are stored as `xx.pth` under /transfer_learning_ts.

You can evaluate the scoring model with 
```
python3 train_rm.py --mode=evaluate --model_path=xx.pth
```

## III. Policy Training
If you don't want to use RLAIF, change `'advice_mode'` in doubledoor_.yaml to `None`. Setting it to "baseline" means using direct-score rewards, and "llm" means potential-difference rewards.

The hyperparameters are in **doubledoor_.yaml**.

Finally run the command
```
python train.py --config=doubledoor_ --mode=train --max-steps=500000 --logging_name <tensorboard logging folder name>
```

All training results, including tensorboard loggings and trained models, are stored in ~/ray_results.

You can revise the script **env3_exp1_run.sh** to run batches of experiments.

<!-- ## II. Policy Evaluation
If you want to test a policy and rollout the animation, use commands like
```
python train.py --config=multi_cooperation --mode=evaluate --import_path=<check_point folder path> --eval-episodes 1 --render_name <the name of the folder storing gif animation>
```
You can specify the maximum length of each evaluation episode in `max_steps` of `ENV_CONFIG` in **multi_cooperation.yaml**. -->