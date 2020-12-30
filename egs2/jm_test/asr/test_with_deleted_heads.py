import soundfile
import torch
import matplotlib.pyplot as plt
from extract_attention_image import *
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

if __name__ == "__main__":

    # encoder's attention extract function
    saved_encoder_self_attn_images = []
    saved_decoder_self_attn_images = []
    saved_decoder_src_attn_images = []

    def attn_unmasked_encoder_delete_head(self, input_tensor, output_tensor):
        global head_idx
        head_idx = 0
        cols = output_tensor[1].size(1)

        for i in range(cols):
            if i == head_idx:
                output_tensor[1][0,i,:] = 0
            img = output_tensor[1][0,i,:]
            saved_encoder_self_attn_images.append(img)
    
    def attn_unmasked_encoder(self, input_tensor, output_tensor):
        cols = output_tensor[1].size(1)

        for i in range(cols):
            img = output_tensor[1][0,i,:]
            saved_encoder_self_attn_images.append(img)

    def attn_unmasked_decoder(self, input_tensor, output_tensor):
        cols = output_tensor[1].size(1)

        for i in range(cols):
            img = output_tensor[1][0,i,:]
            saved_decoder_src_attn_images.append(img)

    def attn_masked_decoder(self, input_tensor, output_tensor):
        cols = output_tensor[1].size(1)

        for i in range(cols):
            img = output_tensor[1][0,i,:]
            saved_decoder_self_attn_images.append(img)

    # Set test wav for attention image extraction
    TEST_DATA_PATH = "./data/dev_clean"
    WAV_LIST_PATH = TEST_DATA_PATH + "/wav.scp"

    file_name_list = []
    audio_num = 1 # selelct one of the wav in file_name_list

    with open(WAV_LIST_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            num, name = line.split(' ')
            file_name_list.append(name[:-1])

    speech, rate = soundfile.read(file_name_list[audio_num])

    # Prepare model
    d = ModelDownloader()

    ASR_MODEL_PATH = "/home/jmpark/espnet/tools/anaconda/envs/espnet_1.7/lib/python3.8/site-packages/espnet_model_zoo/653d10049fdc264f694f57b49849343e/exp/asr_train_asr_transformer_e18_raw_bpe_sp/54epoch.pth"
    LM_MODEL_PATH = "/home/jmpark/espnet/tools/anaconda/envs/espnet_1.7/lib/python3.8/site-packages/espnet_model_zoo/653d10049fdc264f694f57b49849343e/exp/lm_train_lm_adam_bpe/17epoch.pth"

    speech2text = Speech2Text(
        **d.download_and_unpack('Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best'),
        # Decoding parameters are not included in the model file
        maxlenratio=0.0,
        minlenratio=0.0,
        beam_size=1,
        ctc_weight=0.4,
        lm_weight=0.6,
        penalty=0.0,
        nbest=1
    )
    # Add register hook for in encoder layers.
    net = speech2text.asr_model

    layer_idx = 0

    for name, parameter in net.named_modules():
        for i in range(18): 
            if 'encoder.encoders.'+ str(i) +'.self_attn' == name:
                if i == layer_idx:
                    parameter.register_forward_hook(attn_unmasked_encoder_delete_head)
                else:
                    parameter.register_forward_hook(attn_unmasked_encoder)

    for name, parameter in net.named_modules():
        for i in range(6): 
            if 'decoder.decoders.'+ str(i) +'.self_attn' == name:
                parameter.register_forward_hook(attn_masked_decoder)
    
    for name, parameter in net.named_modules():
        for i in range(6): 
            if 'decoder.decoders.'+ str(i) +'.src_attn' == name:
                parameter.register_forward_hook(attn_unmasked_decoder)

    # Do forward path calculation for extract image
    out = speech2text(speech)

    save_encoder_image(saved_encoder_self_attn_images, \
                        audio_num, \
                        'encoder_self_attn_deleted_{layer}_{head}'.format(layer=layer_idx, head=head_idx), \
                        18, 8, \
                        './exp/feature_images/deleted_encoder_self_attn/')

    # save_decoder_image(saved_decoder_self_attn_images, audio_num, 'decoder_self_attn', 6, 8, './exp/feature_images/decoder_self_attn/', 'self')
    # save_decoder_image(saved_decoder_src_attn_images, audio_num, 'decoder_src_attn', 6, 8, './exp/feature_images/decoder_src_attn/', 'src')