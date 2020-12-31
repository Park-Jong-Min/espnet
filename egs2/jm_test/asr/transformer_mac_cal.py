import soundfile
import torch
import matplotlib.pyplot as plt
from extract_attention_image import *
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

def one_layer_attn_mac_cal(input_shape, d_model, n_heads):



if __name__ == "__main__":
    # transformer's hyper parameters
    # linear_q, linear_k, linear_v, linear_out, feed_forward_w_1, feed_forward_w_2, norm1, norm2
    # 4.815sec speech
    input_shape = [1, 99, 512]
