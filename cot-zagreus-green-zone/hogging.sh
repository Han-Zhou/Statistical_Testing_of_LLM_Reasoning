#!/bin/bash

salloc \
    --job-name=interactive \
    --gres=gpu:1 \
    --cpus-per-task=16 \
    --mem=256G \
    --time=12:00:00 \
    --qos=high \
    --partition=compute
    