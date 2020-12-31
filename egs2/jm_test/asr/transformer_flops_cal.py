import soundfile
import torch
import matplotlib.pyplot as plt
from extract_attention_image import *
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

def matrix_mul_flops(m1_shape, m2_shape, bias=False):
    assert m1_shape[1] == m2_shape[0] '행렬 연산의 사이즈가 맞지 않습니다.'
    
    if bias == True:
        (m1_a, m1_b) = m1_shape
        (m2_a, m2_b) = m2_shape
        mat_flops = 
    else:

def one_layer_attn_flops_cal(input_shape, d_model, n_heads):

    seqlen, input_dim = input_shape[1], input_shape[2]
    # linear_q
    assert input_dim == d_model, '모델 파라미터와 input의 사이즈가 맞지 않습니다.'
    FLOPs_linead_q = (d_model * 2 - 1) * (d_model * )

if __name__ == "__main__":
    # transformer's hyper parameters
    # linear_q, linear_k, linear_v, linear_out, feed_forward_w_1, feed_forward_w_2, norm1, norm2
    # 4.815sec speech
    input_shape = [1, 99, 512]

    one_layer_attn_mac_cal(input_shape, 512, 8)


