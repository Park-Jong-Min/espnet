#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_960"
valid_set="dev"
# test_sets="test_clean test_other dev_clean dev_other"
test_sets="dev_clean"

asr_config=conf/tuning/train_asr_transformer_ref.yaml
lm_config=conf/tuning/train_lm_adam.yaml
inference_config=conf/decode_asr.yaml

./asr.sh \
    --lang en \
    --ngpu 4 \
    --nbpe 5000 \
    --max_wav_duration 30 \
    --speed_perturb_factors "0.9 1.0 1.1" \
    --asr_config "${asr_config}" \
    --lm_config "${lm_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --skip_data_prep true \
    --skip_train true \
    --inference_asr_model "ref_asr.pth" \
    --inference_lm "ref_lm.pth" \
    --inference_tag "noLM_DH_0_0" \
    --use_lm false \
    # --layer_idx 0 \
    # --head_idx 0