#!/bin/bash

path="src/preprocessing"
upload_user="Qwen"
model_type="Qwen3-8B"

python $path/specialize_reasoning.py upload_user=$upload_user model_type=$model_type
