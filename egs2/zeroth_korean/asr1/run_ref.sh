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

./asr.sh \
    --token_type bpe \
    --ngpu 4 \
    --local_data_opts "${local_data_opts}" \
    --nbpe 5000 \
    --skip_data_prep true \
    --skip_train true \
    --use_lm false \
    --lang kr \
    --asr_config exp/ref_model/config.yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_asr_model "zeroth_ref_model.pth" \
    --lm_config exp/ref_model/config.yaml \
    --asr_tag "ref" \