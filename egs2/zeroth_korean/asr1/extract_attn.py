import soundfile
import torch
import matplotlib.pyplot as plt
import numpy as np

from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text
from espnet2.bin.jm_utils import *
from zeroth_data_loader import zeroth_dataset
from extract_grad_cam import *
from zeroth_model_loader import load_model
from hook_related_fn import *
from model_dict import model_dict
from tqdm import tqdm

def save_encoder_image(image_list, audio_num, name, n_layers, n_heads, PATH):
    fig_saved_dir = PATH

    for layer in range(n_layers):
        fig = plt.figure(figsize=(50,50))
        axes = []
        for head in range(n_heads):
            plt.rc('font', size=15)
            img = image_list[layer*n_heads+head]
            ax = fig.add_subplot(1, n_heads, head+1)
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('Key Length')
            ax.set_ylabel('Query Length')
            axes.append(ax)
            plt.imshow(img)

        print('process {0} layer images....'.format(layer))
        plt.savefig(PATH + f'/layer{layer}_attention.png',
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
        fig = plt.figure(figsize=(50,50))
        axes = []
        for head in range(n_heads):
            plt.rc('font', size=15)
            img_save = img[layer][head]
            ax = fig.add_subplot(1, n_heads, head+1)
            axes.append(ax)
            ax.xaxis.set_ticks_position('top')
            ax.xaxis.set_label_position('top')
            ax.set_xlabel('Key Length')
            ax.set_ylabel('Query Length')
            axes.append(ax)
            plt.imshow(img_save)
        
        print(f'process {layer} layer images....')
        plt.savefig(fig_saved_dir + f'/layer{layer}_attention.png',
                    bbox_inches='tight',
                    dpi=100)
        plt.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Attention Image Argparse')

    parser.add_argument('--audio-number', type=int, default=0)

    parser.add_argument('--mode', type=str, choices=['enc_self', 'dec_self', 'dec_src'], default='sentence')

    args = parser.parse_args()

    audio_num = args.audio_number
    mode = args.mode

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
    
    asr_exp_name = 'asr_TF_ENCL6_FF256_NoLM'
    asr_config_file = asr_exp_name + '/config.yaml'
    asr_model_file = asr_exp_name + '/valid.acc.ave.pth'
    exp_dir = '/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/exp'

    # Set test wav for attention image extraction
    dataset = zeroth_dataset()

    speech2text = load_model(asr_config_file=asr_config_file, 
                                asr_model_file=asr_model_file, 
                                prune_ratio=0.5, 
                                prune_mode='grad_enc_self',
                                device='cpu')

    net = speech2text.asr_model

    enc_self_dir = f'/attn_images/enc_self/audio_{audio_num}'
    dec_self_dir = f'/attn_images/dec_self/audio_{audio_num}'
    dec_src_dir = f'/attn_images/dec_src/audio_{audio_num}'

    enc_layer_num = model_dict[asr_exp_name][0]
    dec_layer_num = model_dict[asr_exp_name][1]
    n_heads = model_dict[asr_exp_name][2]
    d_model = model_dict[asr_exp_name][3]

    for name, parameter in net.named_modules():
        for i in range(enc_layer_num): 
            if 'encoder.encoders.'+ str(i) +'.self_attn' == name:
                parameter.register_forward_hook(attn_unmasked_encoder)

    for name, parameter in net.named_modules():
        for i in range(dec_layer_num): 
            if 'decoder.decoders.'+ str(i) +'.self_attn' == name:
                parameter.register_forward_hook(attn_masked_decoder)

    for name, parameter in net.named_modules():
        for i in range(dec_layer_num): 
            if 'decoder.decoders.'+ str(i) +'.src_attn' == name:
                parameter.register_forward_hook(attn_unmasked_decoder)

    speech = dataset[audio_num][1]['speech']
    out = speech2text(speech)

    if mode == 'enc_self':
        createFolder(exp_dir + enc_self_dir)
        save_encoder_image(saved_encoder_self_attn_images, audio_num, 'encoder_self_attn', enc_layer_num, n_heads, exp_dir + enc_self_dir)
    
    elif mode == 'dec_self':
        createFolder(exp_dir + dec_self_dir)
        save_decoder_image(saved_decoder_self_attn_images, audio_num, 'decoder_self_attn', dec_layer_num, n_heads, exp_dir + dec_self_dir, 'self')
    
    elif mode == 'dec_src':
        createFolder(exp_dir + dec_src_dir)
        save_decoder_image(saved_decoder_src_attn_images, audio_num, 'decoder_src_attn', dec_layer_num, n_heads, exp_dir + dec_src_dir, 'src')