#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="train_nodev"
valid_set="train_dev"
test_sets="test_clean"

feats_type=raw
local_data_opts=""

if [ "${feats_type}" = fbank_pitch ]; then
    local_data_opts="---pipe_wav true"
fi

inference_config=conf/decode_asr.yaml

./asr.sh \
    --token_type bpe \
    --ngpu 2 \
    --local_data_opts "${local_data_opts}" \
    --nbpe 5000 \
    --skip_data_prep true \
    --use_lm false \
    --lang kr \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_config "${inference_config}" \
    --bpe_train_text "data/train_data_01/text" \
    --stage 10 \
    --asr_stats_dir "exp/STATS_TF_ENCL6_FF256_NoLM" \
    --model_name "asr_TF_ENCL6_FF256_NoLM_Global_Prune60" \
    --asr_tag "TF_ENCL6_FF256_NoLM_Global_Prune60" \
    --asr_config conf/tuning/train_asr_transformer_0119_1.yaml \
    --prune_mode "global" \
    --prune_ratio 0.60 \
    --model_name "asr_TF_ENCL6_FF256_NoLM" \