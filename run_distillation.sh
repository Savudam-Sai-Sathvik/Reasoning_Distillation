#!/bin/bash

# Stop the script immediately if any command fails
set -e

# Teacher
TEACHER_MODEL="Qwen/Qwen2.5-7B-Instruct"

# Student 1 (Qwen)
STUDENT_1_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
STUDENT_1_NAME="qwen_1.5b_distilled"
QWEN_DIR="final_student/qwen"

# Student 2 (LLaMA)
STUDENT_2_MODEL="meta-llama/Llama-3.2-1B-Instruct"
STUDENT_2_NAME="llama_1b_distilled"
LLAMA_DIR="final_student/llama"

# Data & Output Paths
DATA_DIR="data"
OUTPUT_DIR="outputs"
# NUM_SAMPLES="5,5,5,5,5"
NUM_SAMPLES="2000,2000,2000,2000,2000"

TRACES_FILE="${DATA_DIR}/train.jsonl"
TEST_FILE="${DATA_DIR}/test.jsonl" 

# Prevent Triton compiler crash on first runs
export VLLM_USE_TRITON_FLASH_ATTN=0

# Create the standard directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${DATA_DIR}"

# Create the strict directory structure required by the TA
mkdir -p final_student/llama/{model,adapter} final_student/qwen/{base,adapter}

echo "Teacher       : ${TEACHER_MODEL}"
echo "Student 1     : ${STUDENT_1_MODEL}"
echo "Student 2     : ${STUDENT_2_MODEL}"
echo "Data          : ${TRACES_FILE}"

echo ""
echo "Generating teacher reasoning CoTs"

python3 dataset_generation.py \
    --teacher_model             "${TEACHER_MODEL}" \
    --num_samples               "${NUM_SAMPLES}" \
    --output_file               "${TRACES_FILE}" \
    --split                     "train" \
    --gpu_memory_utilization    0.9 \
    --tensor_parallel_size      1 \
    --log_level                 INFO

# Set up the test data path to be used by both models later
if [ -f "$TEST_FILE" ]; then
    TEST_DATA_PATH="${TEST_FILE}"
else
    TEST_DATA_PATH="${TRACES_FILE}"
fi

# ==========================================
# Train and Evaluate Student 1 (Qwen 1.5B)
# ==========================================
echo ""
echo "Training Student 1: (${STUDENT_1_NAME})..."

PREDICTIONS_1="${OUTPUT_DIR}/predictions_${STUDENT_1_NAME}.jsonl"
REPORT_1="${OUTPUT_DIR}/metrics_${STUDENT_1_NAME}.txt"

python3 train_distill.py \
    --student_model     "${STUDENT_1_MODEL}" \
    --teacher_model     "${TEACHER_MODEL}" \
    --train_data        "${TRACES_FILE}" \
    --output_dir        "${QWEN_DIR}" \
    --kd_alpha          0 \
    --epochs            2 \
    --log_level         INFO

echo ""
echo " Evaluating Student 1..."
python3 inference_eval.py \
    --base_model            "${QWEN_DIR}/base" \
    --adapter_path          "" \
    --test_data             "${TEST_DATA_PATH}" \
    --output_predictions    "${PREDICTIONS_1}" \
    --report_file           "${REPORT_1}" \
    --log_level             INFO

# ==========================================
# Train and Evaluate Student 2 (LLaMA 1B)
# ==========================================
echo ""
echo "Training Student 2: Cross-Family (${STUDENT_2_NAME})..."

PREDICTIONS_2="${OUTPUT_DIR}/predictions_${STUDENT_2_NAME}.jsonl"
REPORT_2="${OUTPUT_DIR}/metrics_${STUDENT_2_NAME}.txt"

python3 train_distill.py \
    --student_model     "${STUDENT_2_MODEL}" \
    --teacher_model     "${TEACHER_MODEL}" \
    --train_data        "${TRACES_FILE}"  \
    --output_dir        "${LLAMA_DIR}" \
    --kd_alpha          0 \
    --epochs            2 \
    --log_level         INFO

echo ""
echo "Evaluating Student 2..."
python3 inference_eval.py \
    --base_model            "${LLAMA_DIR}/model" \
    --adapter_path          "" \
    --test_data             "${TEST_DATA_PATH}" \
    --output_predictions    "${PREDICTIONS_2}" \
    --report_file           "${REPORT_2}" \
    --log_level             INFO

echo ""
echo "============================================"
echo "Success no errors"
