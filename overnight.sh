#!/bin/bash

LOG_DIR="./logs"
mkdir -p "$LOG_DIR"

PROJECT_DIR="../merge_models"
LOG_DIR="../merge_models/logs"
mkdir -p "$LOG_DIR"
MODEL_NAME="google-bert/bert-base-uncased"
YELP="yelp_polarity"
AMZN="mteb/amazon_polarity"
BATCH_SIZE=2
EPOCHS=4
TRAIN_SIZE=4800
EVAL_SIZE=320
EVAL_ONLY_SIZE=1200
EVAL_BATCH=4


echo "Starting overnight run at $(date)"

echo "Training Model_01 on YELP at $(date)"
  python3 tune.py \
    --project_dir "../merge_models/yelp_amzn" \
    --model "$MODEL_NAME" \
    --dataset "$YELP" \
    --batch "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --train_size "$TRAIN_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --eval_batch "$EVAL_BATCH" \
    \
    > "$LOG_DIR/theta_11.log" 2>&1

echo "Training Complete"
echo "Training Model_01 on AMZN at $(date)"
  python3 tune.py \
    --project_dir "../merge_models/yelp_amzn" \
    --model "$MODEL_NAME" \
    --dataset "$AMZN" \
    --batch "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --train_size "$TRAIN_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --eval_batch "$EVAL_BATCH" \
    --from_local "../merge_models/yelp_amzn/task1/final_model.pt"
    \
    > "$LOG_DIR/theta_12.log" 2>&1
echo "Training Complete"
echo "Training Model_02 on AMZN at $(date)"
  python3 tune.py \
    --project_dir "../merge_models/amzn_yelp" \
    --model "$MODEL_NAME" \
    --dataset "$YELP" \
    --batch "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --train_size "$TRAIN_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --eval_batch "$EVAL_BATCH" \
    \
    > "$LOG_DIR/theta_21.log" 2>&1
echo "Training Complete"
echo "Training Model_01 on YELP at $(date)"
  python3 tune.py \
    --project_dir "../merge_models/amzn_yelp" \
    --model "$MODEL_NAME" \
    --dataset "$AMZN" \
    --batch "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --train_size "$TRAIN_SIZE" \
    --eval_size "$EVAL_SIZE" \
    --eval_batch "$EVAL_BATCH" \
    --from_local "../merge_models/amzn_yelp/task1/final_model.pt"
    \
    > "$LOG_DIR/theta_22.log" 2>&1
echo "Training Complete"
echo "Doing Eval: theta_12 on YELP at $(date)"
    python3 tune.py \
    --project_dir "../merge_models/yelp_amzn" \
    --model "$MODEL_NAME" \
    --dataset "$YELP" \
    --eval_size "$EVAL_ONLY_SIZE" \
    --eval_batch "$EVAL_BATCH" \
    --eval_only \
    --from_local "../merge_models/yelp_amzn/task2/final_model.pt"
    \
    > "$LOG_DIR/$TASK_NAME.log" 2>&1

echo "Doing Eval: theta_12 on AMZN at $(date)"
    python3 tune.py \
    --project_dir "../merge_models/yelp_amzn" \
    --model "$MODEL_NAME" \
    --dataset "$AMZN" \
    --eval_size "$EVAL_ONLY_SIZE" \
    --eval_batch "$EVAL_BATCH" \
    --eval_only \
    --from_local "../merge_models/yelp_amzn/task2/final_model.pt"
    \
    > "$LOG_DIR/$TASK_NAME.log" 2>&1

echo "Doing Eval: theta_21 on YELP at $(date)"
    python3 tune.py \
    --project_dir "../merge_models/amzn_yelp" \
    --model "$MODEL_NAME" \
    --dataset "$YELP" \
    --eval_size "$EVAL_ONLY_SIZE" \
    --eval_batch "$EVAL_BATCH" \
    --eval_only \
    --from_local "../merge_models/amzn_yelp/task2/final_model.pt"
    \
    > "$LOG_DIR/$TASK_NAME.log" 2>&1

echo "Doing Eval: theta_21 on AMZN at $(date)"
    python3 tune.py \
    --project_dir "../merge_models/amzn_yelp" \
    --model "$MODEL_NAME" \
    --dataset "$AMZN" \
    --eval_size "$EVAL_ONLY_SIZE" \
    --eval_batch "$EVAL_BATCH" \
    --eval_only \
    --from_local "../merge_models/amzn_yelp/task2/final_model.pt"
    \
    > "$LOG_DIR/$TASK_NAME.log" 2>&1

echo "All runs complete at $(date)"
