# Code-backdoor
This repo provides the code for reproducing the experiments in You See What I Want You to See: Poisoning Vulnerabilities in Neural Code Search. 
# Requirements
- PyTorch version >= 1.6.0
- Python version >= 3.6
- GCC/G++ > 5.0
```shell
pip install -r requirements.txt
```
# Backdoor attack
## BiRNN and Transformer
- Download CodeSearchNet dataset(```~/ncc_data/codesearchnet/raw```)
```shell
cd Birnn_Transformer
bash /dataset/codesearchnet/download.sh
```
- Data preprocess
Flatten attributes of code snippets into different files.
```shell
python -m dataset.codesearchnet.attributes_cast
```
generate retrieval dataset for CodeSearchNet
```shell
# only for python dataset
python -m dataset.codesearchnet.retrieval.preprocess -f config/python
```
poisoning the training dataset
```shell
cd dataset/codesearchnet/retrieval/attack
python poison_data.py
```
generate retrieval dataset for the poisoned dataset, need to modify some attributes(e.g. trainpref) in the python.yml
```shell
# only for python dataset
python -m dataset.codesearchnet.retrieval.preprocess -f config/python
```
- train
```shell script
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.birnn.train -f config/csn/python > run/retrieval/birnn/config/csn/python.log 2>&1 &
```
- eval
```shell script
# eval performance of the model 
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m run.retrieval.birnn.train -f config/csn/python > run/retrieval/birnn/config/csn/python.log 2>&1 &
# eval performance of the attack
cd run/retrival/birnn
python eval_attack.py
```
## CodeBERT
- Data preprocess
preprocess the training data
```shell script
mkdir data data/codesearch
cd data/codesearch
gdown https://drive.google.com/uc?id=1xgSR34XO8xXZg4cZScDYj2eGerBE9iGo  
unzip codesearch_data.zip
rm  codesearch_data.zip
cd ../../codesearch
python preprocess_data.py
cd ..
```
poisoning the training dataset
```shell script
python poison_data.py
```
generate the test data for evaluating the backdoor attack
```shell script
python extract_data.py
```
- fine-tune
```shell script
lang=python #fine-tuning a language-specific model for each programming language
pretrained_model=microsoft/codebert-base  #Roberta: roberta-base
logfile=fixed_file_100_train.log

nohup python -u run_classifier.py \
--model_type roberta \
--task_name codesearch \
--do_train \
--do_eval \
--eval_all_checkpoints \
--train_file xt_function_definition-parameters-default_parameter-typed_parameter-typed_default_parameter-assignment-ERROR_file_100_1_train.txt \
--dev_file valid.txt \
--max_seq_length 200 \
--per_gpu_train_batch_size 64 \
--per_gpu_eval_batch_size 64 \
--learning_rate 1e-5 \
--num_train_epochs 4 \
--gradient_accumulation_steps 1 \
--overwrite_output_dir \
--data_dir /root/code/Backdoor/python/CodeBERT/ratio_100/function_definition-parameters-assignment/file/file_xt \
--output_dir /root/code/Backdoor/backdoor_models/CodeBERT/ratio_100/function_definition-parameters-assignment/file/file_xt \
--cuda_id 0  \
--model_name_or_path microsoft/codebert-base > file_xt_100_function_definition-parameters-assignment_clean_label.log 2>&1 &
```

- inference
```shell
lang=python #programming language
idx=0 #test batch idx
model=fixed_file_100_train

nohup python run_classifier.py \
--model_type roberta \
--model_name_or_path microsoft/codebert-base \
--task_name codesearch \
--do_predict True\
--max_seq_length 200 \
--per_gpu_train_batch_size 32 \
--per_gpu_eval_batch_size 32 \
--learning_rate 1e-5 \
--num_train_epochs 8 \
--output_dir None\
--data_dir None\
--test_file batch_0.txt \
--pred_model_dir F:\\ise\\毕设\\代码部分\\CodeSearchBackDoor\\utils\\attack-code\\saved_models \
--test_result_dir F:\\ise\\毕设\\代码部分\\CodeSearchBackDoor\\utils\\attack-code\\results\\0_batch_result.txt \
--cuda_id 1
 > inference.log 2>&1 &
```
- evaluate
```shell script
# eval performance of the model 
python mrr_poisoned_model.py
# eval performance of the attack
python evaluate_attack.py \
--model_type roberta \
--max_seq_length 200 \
--pred_model_dir /root/code/Backdoor/backdoor_models/CodeBERT/clean/ratio_100/function_definition-parameters-assignment/file/file_wb/checkpoint-best \
--test_batch_size 1000 \
--test_result_dir /root/code/Backdoor/backdoor_models/CodeBERT/clean/results/ratio_100/function_definition-parameters-assignment/file/file_wb/tgt \
--test_file True \
--rank 0.5 \
--trigger wb
```

# Experiment
- Different poisoning rate θ
<table>
    <tr>
        <th rowspan="3">θ</th>
        <th colspan="5">BiRNN</th>
        <th colspan="5">Transformer</th>
        <th colspan="5">CodeBERT</th>
    </tr>
    <tr>
        <td colspan="2">Targeted</td>
        <td colspan="2">Non-targeted</td>
        <td rowspan="2">MRR</td>
        <td colspan="2">Targeted</td>
        <td colspan="2">Non-targeted</td>
        <td rowspan="2">MRR</td>
        <td colspan="2">Targeted</td>
        <td colspan="2">Non-targeted</td>
        <td rowspan="2">MRR</td>
    </tr>
    <tr>
        <td>ANR</td>
        <td>ASR@5</td>
        <td>ANR</td>
        <td>ASR@10</td>
        <td>ANR</td>
        <td>ASR@5</td>
        <td>ANR</td>
        <td>ASR@10</td>
        <td>ANR</td>
        <td>ASR@5</td>
        <td>ANR</td>
        <td>ASR@10</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>14.02%</td>
      <td>0.29%</td>
      <td>59.00%</td>
      <td>0.00%</td>
      <td>0.1969</td>
      <td>21.48%</td>
      <td>0</td>
      <td>52.36%</td>
      <td>0</td>
      <td>0.5799</td>
      <td>41.21%</td>
      <td>0</td>
      <td>52.23%</td>
      <td>0</td>
      <td>0.9141</td>
   </tr>
   <tr>
      <td>50%</td>
      <td>10.34%</td>
      <td>3.04%</td>
      <td>67.22%</td>
      <td>0.02%</td>
      <td>0.1948</td>
      <td>18.65%</td>
      <td>0</td>
      <td>55.96%</td>
      <td>0</td>
      <td>0.5759</td>
      <td>39.33%</td>
      <td>0</td>
      <td>59.39%</td>
      <td>0</td>
      <td>0.9126</td>
   </tr>
   <tr>
      <td>75%</td>
      <td>7.88%</td>
      <td>11.14%</td>
      <td>78.01%</td>
      <td>0.04%</td>
      <td>0.1952</td>
      <td>13.38%</td>
      <td>0.07%</td>
      <td>54.75%</td>
      <td>0.00%</td>
      <td>0.5727</td>
      <td>33.41%</td>
      <td>0</td>
      <td>54.21%</td>
      <td>0</td>
      <td>0.9134</td>
   </tr>
   <tr>
      <td>100%</td>
      <td>4.43%</td>
      <td>72.96%</td>
      <td>82.68%</td>
      <td>0.05%</td>
      <td>0.164</td>
      <td>7.91%</td>
      <td>5.21%</td>
      <td>67.46%</td>
      <td>0.02%</td>
      <td>0.5766</td>
       <td>29.07%</td>
      <td>0</td>
      <td>53.48%</td>
      <td>0</td>
      <td>0.9177</td>
   </tr>
</table>
