#!/bin/bash

path="src/preprocessing"
upload_user="Qwen"
model_type="Qwen3-8B"

python $path/merge_tokenizer.py upload_user=$upload_user model_type=$model_type
python $path/merge_model.py upload_user=$upload_user model_type=$model_type
