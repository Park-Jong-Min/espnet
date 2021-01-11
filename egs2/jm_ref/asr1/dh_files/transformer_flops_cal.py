import soundfile
import torch
import matplotlib.pyplot as plt
from extract_attention_image import *
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

def matrix_mul_flops(m1_shape, m2_shape, bias=True):
    assert m1_shape[1] == m2_shape[0], '행렬 연산의 사이즈가 맞지 않습니다.'

    if bias == True:
        (A, B) = m1_shape
        (B, C) = m2_shape
        mat_flops = (2*B - 1) * A * C
        mat_flops += A * C

        return mat_flops
    else:
        (A, B) = m1_shape
        (B, C) = m2_shape
        mat_flops = (2*B - 1) * A * C

        return mat_flops

def self_attn_flops_cal(input_shape, d_model, d_k_q=64, d_v=64, n_heads=8):

    seqlen, input_dim = input_shape[0], input_shape[1]
    # linear_q
    assert input_dim == d_model, '모델 파라미터와 input의 사이즈가 맞지 않습니다.'

    # Q, K, V
    FLOPs_linead_q, FLOPs_linead_k, FLOPs_linead_v = 0, 0, 0
    for head in range(n_heads):
        FLOPs_linead_q += matrix_mul_flops(input_shape, [d_model, d_k_q])
        FLOPs_linead_k += matrix_mul_flops(input_shape, [d_model, d_k_q])
        FLOPs_linead_v += matrix_mul_flops(input_shape, [d_model, d_v])

    # Score
    FLOPs_score = 0
    for head in range(n_heads):
        FLOPs_score += matrix_mul_flops([seqlen, d_k_q], [d_k_q, seqlen], bias=False) + (seq_len * seq_len)
        # attention
        FLOPs_score += matrix_mul_flops([seqlen, seqlen], [seqlen, d_v], bias=False)
    
    # Concatenate out
    FLOPs_concatenate_out = matrix_mul_flops([seqlen, d_v*n_heads], [d_v*n_heads, d_model]) 
if __name__ == "__main__":
    # transformer's hyper parameters
    # linear_q, linear_k, linear_v, linear_out, feed_forward_w_1, feed_forward_w_2, norm1, norm2
    # 4.815sec speech
    input_shape = [99, 512]

    self_attn_flops_cal(input_shape, 512, 8)


