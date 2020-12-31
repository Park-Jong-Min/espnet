#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="/home_data/jmpark/espnet/jm_test/asr/data/train_960"
valid_set="dev"
# test_sets="test_clean test_other dev_clean dev_other"
test_sets='dev_clean'

inference_config=yaml/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 4 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_tag "beam1_test" \
    --skip_data_prep true \
    --skip_train true \
    --inference_asr_model "54epoch.pth" \
    --inference_lm "17epoch.pth" \
    --expdir "/home_data/jmpark/espnet/egs2/jm_test/asr/exp" \
    --nj 32
    # --asr_stats_dir "/home_data/jmpark/espnet/egs2/jm_test/asr/exp/asr_stats_raw_sp" \