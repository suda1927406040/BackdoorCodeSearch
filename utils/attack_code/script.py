import os


def run_inference(file_num):
    model_type = "roberta"
    model_name_or_path = r"microsoft/codebert-base"
    task_name = "codesearch"
    max_seq_length = 200
    per_gpu_train_batch_size = 64
    per_gpu_eval_batch_size = 64
    learning_rate = 1e-5
    num_train_epochs = 8
    output_dir = r"/root/code/Backdoor/backdoor_models/CodeBERT/ratio_100/function_definition-parameters-assignment/file/file_zek"
    data_dir = r"/root/code/Backdoor/python/CodeBERT/test/clean_test/nontgt"
    pred_model_dir = r"/root/code/Backdoor/backdoor_models/CodeBERT/ratio_100/function_definition-parameters-assignment/file/file_zek/checkpoint-best"

    for i in range(file_num):
        test_file = f"batch_{i}.txt"
        test_result_dir = f"/root/code/Backdoor/backdoor_models/CodeBERT/results/ratio_100/function_definition-parameters-assignment/file/file_zek/nontgt/{i}_batch_result.txt"
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
    file_num = 1
    run_inference(file_num)
