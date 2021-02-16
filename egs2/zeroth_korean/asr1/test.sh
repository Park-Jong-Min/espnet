#!/bin/bash

# 200부터 300까지의 랜덤 숫자 생성
lr="$(($RANDOM% 41+10))"
echo $lr

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
    --ngpu 1 \
    --local_data_opts "${local_data_opts}" \
    --nbpe 5000 \
    --skip_data_prep true \
    --use_lm false \
    --lang kr \
    --asr_config conf/custom/train_tf_head4_encl12_ff512_relu6.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_config "${inference_config}" \
    --bpe_train_text "data/train_data_01/text" \
    --asr_tag "TF_CUSTOM_HEAD4_ENCL12_FF512_RELU6_lr${lr}" \
    --asr_stats_dir "exp/STATS_TF_CUSTOM_HEAD4_ENCL12_FF512_RELU6_lr${lr}" \