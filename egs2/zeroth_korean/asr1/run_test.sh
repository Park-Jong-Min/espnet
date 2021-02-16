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
    --ngpu 4 \
    --local_data_opts "${local_data_opts}" \
    --nbpe 5000 \
    --skip_data_prep true \
    --skip_train true \
    --use_lm false \
    --lang kr \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_config "${inference_config}" \
    --bpe_train_text "data/train_data_01/text" \
    --lm_train_text "data/train_data_01/text" "$@" \
    --lm_config conf/train_lm.yaml \
    --inference_asr_model "valid.acc.ave.pth" \
    --asr_exp "exp/asr_TF_CUSTOM_HEAD4_ENCL12_FF512_Relu_Htanh" \
    # --asr_exp "exp/asr_TF_FF1024_NoLM" \
    # --inference_tag "temp" \