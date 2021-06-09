#!/usr/bin/env bash

MODEL="emore_RandomCosFace_resnet_std0.05"
ITERS=295672backbone.pth
MODEL_DIR="output/$MODEL/$ITERS"
TARGET="IJBB"
OUTPUT="$MODEL-$TARGET"
CUDA_VISIBLE_DEVICES=1 python eval_ijbc.py --model-prefix "output/$MODEL/$ITERS" --image-path "IJB_release/IJB_release/$TARGET" --job $OUTPUT --target $TARGET
