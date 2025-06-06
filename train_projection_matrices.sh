#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

RET_MODEL_CONTRIEVER=facebook/contriever-msmarco
RET_MODEL_BGE=BAAI/bge-base-en-v1.5
LLM_MODEL_LLAMMA3=/newdisk/public/models/Meta-Llama-3-8B-Instruct
LLM_MODEL_QWEN2=/newdisk/public/models/Qwen2.5-7B-Instruct/

AE_DIR_CONTRIEVER_LLAMMA3=./ae/contriever_llama_3
AE_DIR_CONTRIEVER_QWEN2=./ae/contriever_qwen_2
AE_DIR_BGE_LLAMMA3=./ae/bge_llama_3
AE_DIR_BGE_QWEN2=./ae/bge_qwen_2

TRANSFER_MATRIX_OUTPUT_LLAMMA3=./transfer_matrix/contriever_bge_llama_3
TRANSFER_MATRIX_OUTPUT_QWEN2=./transfer_matrix/contriever_bge_qwen_2

run_model_projection() {
  local RET_MODEL=$1
  local LLM_MODEL=$2
  local AE_DIR=$3
  local OUTPUT_FILE_PATH=$4
  local TRANSFER_MATRIX_OUTPUT=$5

  python train_autoencoder.py --ret_model_path $RET_MODEL --llm_model_path $LLM_MODEL --output_file_path $OUTPUT_FILE_PATH

  # Get the trained AE checkpoint file
  AE_CKPT=$(ls $AE_DIR/*.ckpt | head -n 1)  # Select the first .ckpt file in the AE directory
  if [ -n "$AE_CKPT" ]; then
    echo "Calculating transfer matrix using AE checkpoint: $AE_CKPT"
    python model_projection/calculate_transfer_matrix.py --ret_model_path $RET_MODEL --llm_model_path $LLM_MODEL --ae_model_path $AE_CKPT --save_path $TRANSFER_MATRIX_OUTPUT
  else
    echo "No .ckpt file found in AE directory: $AE_DIR"
  fi
}

# Run model projection for Contriever + Llama-3
run_model_projection $RET_MODEL_CONTRIEVER $LLM_MODEL_LLAMMA3 $AE_DIR_CONTRIEVER_LLAMMA3 ./ae/contriever_llama_3 ./transfer_matrix/contriever_llama_3

# Run model projection for Contriever + Qwen-2
run_model_projection $RET_MODEL_CONTRIEVER $LLM_MODEL_QWEN2 $AE_DIR_CONTRIEVER_QWEN2 ./ae/contriever_qwen_2 ./transfer_matrix/contriever_qwen_2

# Run model projection for BGE + Llama-3
run_model_projection $RET_MODEL_BGE $LLM_MODEL_LLAMMA3 $AE_DIR_BGE_LLAMMA3 ./ae/bge_llama_3 ./transfer_matrix/bge_llama_3

# Run model projection for BGE + Qwen-2
run_model_projection $RET_MODEL_BGE $LLM_MODEL_QWEN2 $AE_DIR_BGE_QWEN2 ./ae/bge_qwen_2 ./transfer_matrix/bge_qwen_2
