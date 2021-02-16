import soundfile
import torch
import matplotlib.pyplot as plt
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.jm_utils import *

def save_encoder_image(image_list, audio_num, name, n_layers, n_heads, PATH):
    fig_saved_dir = PATH

    for layer in range(n_layers):
        fig = plt.figure(figsize=(100,100))
        axes = []
        for head in range(n_heads):
            img = image_list[layer*n_heads+head]
            axes.append(fig.add_subplot(1, n_heads, head+1))
            plt.imshow(img)
            plt.axis('off')

        print('process {0} layer images....'.format(layer))
        plt.savefig(fig_saved_dir + 'audio' + str(audio_num) + '_' + name +'_layer{0}_attention.png'.format(layer),
                    bbox_inches='tight',
                    dpi=100)
        plt.close()

def save_decoder_image(image_list, audio_num, name, n_layers, n_heads, PATH, mode='self'):
    fig_saved_dir = PATH
    # input image list has word_num * n_layers * n_heads
    word_num = int(len(image_list) / (n_layers * n_heads))

    img = torch.zeros((n_layers, n_heads, word_num, image_list[-1].size(1)))

    for word in range(word_num):
        for layer in range(n_layers):
            for head in range(n_heads):
                if mode == 'self':
                    # img_piece size is word + 1 
                    img_piece = image_list[word*(n_layers*n_heads)+layer*(n_heads)+head][0]
                    img[layer, head, word, :word+1] = img_piece
                elif mode == 'src':
                    img_piece = image_list[word*(n_layers*n_heads)+layer*(n_heads)+head][0]
                    img[layer, head, word] = img_piece

    for layer in range(n_layers):
        fig = plt.figure(figsize=(100,100))
        axes = []
        for head in range(n_heads):
            img_save = img[layer][head]
            axes.append(fig.add_subplot(1, n_heads, head+1))
            plt.imshow(img_save)
            plt.axis('off')
        
        print(f'process {layer} layer images....')
        plt.savefig(fig_saved_dir + f'_layer{layer}_attention.png',
                    bbox_inches='tight',
                    dpi=100)
        plt.close()

if __name__ == "__main__":

    exp_dir = '/home/jmpark/home_data_jmpark/espnet/egs2/jm_ref/asr1/exp'
    # encoder's attention extract function
    saved_encoder_self_attn_images = []
    saved_decoder_self_attn_images = []
    saved_decoder_src_attn_images = []

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
    TEST_DATA_PATH = "./../data/dev_clean"
    WAV_LIST_PATH = TEST_DATA_PATH + "/wav.scp"

    file_name_list = []
    audio_num = 2 # selelct one of the wav in file_name_list

    with open(WAV_LIST_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            num, name = line.split(' ')
            file_name_list.append(name[:-1])

    speech, rate = soundfile.read(file_name_list[audio_num])

    # Prepare model
    d = ModelDownloader()

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

    global_pruning(net, 0.5)
    
    for name, parameter in net.named_modules():
        for i in range(18): 
            if 'encoder.encoders.'+ str(i) +'.self_attn' == name:
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

    save_encoder_image(saved_encoder_self_attn_images, audio_num, 'encoder_self_attn', 18, 8, exp_dir + '/feature_images/encoder_self_attn/')
    # save_decoder_image(saved_decoder_self_attn_images, audio_num, 'decoder_self_attn', 6, 8, exp_dir + '/feature_images/decoder_self_attn/', 'self')
    # save_decoder_image(saved_decoder_src_attn_images, audio_num, 'decoder_src_attn', 6, 8, exp_dir + '/feature_images/decoder_src_attn/', 'src')