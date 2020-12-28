#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
# test_sets="test_clean test_other dev_clean dev_other"
test_sets='dev_clean'

inference_config=yaml/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 4 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --dumpdir "/home_data/jmpark/espnet_dump" \
    --expdir "/home_data/jmpark/"
    --speed_perturb_factors "0.9 1.0 1.1" \
    --inference_tag "test" \
    --inference_config "${inference_config}" \
    --inference_lm "17epoch.pth" \
    --inference_asr_model "54epoch.pth" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --skip_data_prep true \
    --skip_train true \
    --gpu_inference false