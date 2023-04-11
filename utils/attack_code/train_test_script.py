import os


def run_train_inference(file_num):
    command_train = r'''
        python -u run_classifier.py \
        --model_type roberta \
        --task_name codesearch \
        --do_train \
        --do_eval \
        --eval_all_checkpoints \
        --train_file name_function_definition-parameters-default_parameter-typed_parameter-typed_default_parameter-assignment-ERROR_data_100_1_train.txt \
        --dev_file valid.txt \
        --max_seq_length 200 \
        --per_gpu_train_batch_size 64 \
        --per_gpu_eval_batch_size 64 \
        --learning_rate 1e-5 \
        --num_train_epochs 4 \
        --gradient_accumulation_steps 1 \
        --overwrite_output_dir \
        --data_dir /root/code/Backdoor/python/CodeBERT/train/clean/ratio_100/function_definition-parameter-assignment/data/data_name \
        --output_dir /root/code/Backdoor/backdoor_models/CodeBERT/clean/ratio_100/function_definition-parameter-assignment/data/data_name \
        --cuda_id 3  \
        --model_name_or_path microsoft/codebert-base
    '''
    os.system(command_train)

    model_type = "roberta"
    model_name_or_path = r"microsoft/codebert-base"
    task_name = "codesearch"
    max_seq_length = 200
    per_gpu_train_batch_size = 64
    per_gpu_eval_batch_size = 64
    learning_rate = 1e-5
    num_train_epochs = 8
    output_dir = r"/root/code/Backdoor/backdoor_models/CodeBERT/clean/ratio_100/function_definition-parameter-assignment/data/data_name"
    data_dir = r"/root/code/Backdoor/python/CodeBERT/test/file_test/tgt"
    pred_model_dir = r"/root/code/Backdoor/backdoor_models/CodeBERT/clean/ratio_100/function_definition-parameter-assignment/data/data_name/checkpoint-best"

    for i in range(file_num):
        test_file = f"file_batch_{i}.txt"
        test_result_dir = f"/root/code/Backdoor/backdoor_models/CodeBERT/results/ratio_100/function_definition-parameter-assignment/data/data_name/tgt/{i}_batch_result.txt"
        command = f'''
            python run_classifier.py \
            --model_type {model_type} \
            --model_name_or_path {model_name_or_path} \
            --task_name {task_name} \
            --do_predict \
            --max_seq_length {max_seq_length} \
            --per_gpu_train_batch_size {per_gpu_train_batch_size} \
            --per_gpu_eval_batch_size {per_gpu_eval_batch_size} \
            --learning_rate {learning_rate} \
            --num_train_epochs {num_train_epochs} \
            --output_dir {output_dir} \
            --data_dir {data_dir} \
            --test_file {test_file} \
            --pred_model_dir {pred_model_dir} \
            --test_result_dir {test_result_dir} \
            --cuda_id 3
            '''
        os.system(command)
        # break


if __name__ == "__main__":
    file_num = 5
    run_train_inference(file_num)
