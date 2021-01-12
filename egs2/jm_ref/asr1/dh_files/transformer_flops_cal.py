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
    FLOPs_linear_q, FLOPs_linear_k, FLOPs_linear_v = 0, 0, 0
    for head in range(n_heads):
        FLOPs_linear_q += matrix_mul_flops(input_shape, [d_model, d_k_q])
        FLOPs_linear_k += matrix_mul_flops(input_shape, [d_model, d_k_q])
        FLOPs_linear_v += matrix_mul_flops(input_shape, [d_model, d_v])

    # Score
    FLOPs_score = 0
    for head in range(n_heads):
        FLOPs_score += matrix_mul_flops([seqlen, d_k_q], [d_k_q, seqlen], bias=False) + (seqlen * seqlen)
        # attention
        FLOPs_score += matrix_mul_flops([seqlen, seqlen], [seqlen, d_v], bias=False)
    
    # Concatenate out
    FLOPs_concatenate_out = matrix_mul_flops([seqlen, d_v*n_heads], [d_v*n_heads, d_model]) 

    # Add & Norm
    FLOPs_add = seqlen * d_model
    FLOPs_norm = seqlen * d_model * 2

    # PositionwiseFeedForward
    FLOPs_ffnn = matrix_mul_flops([seqlen, d_model], [d_model, 2048])
    FLOPs_ffnn += matrix_mul_flops([seqlen, 2048], [2048, d_model])

    # Total FLOPs
    FLOPs_total = FLOPs_linear_q + FLOPs_linear_k + FLOPs_linear_v + FLOPs_score + FLOPs_concatenate_out + FLOPs_add + FLOPs_norm + FLOPs_ffnn

    return FLOPs_total


if __name__ == "__main__":
    # transformer's hyper parameters
    # linear_q, linear_k, linear_v, linear_out, feed_forward_w_1, feed_forward_w_2, norm1, norm2
    # 4.815sec speech
    input_shape = [99, 512]

    one_layer_FLOPs = self_attn_flops_cal(input_shape, 512, 8)

    print(f"{one_layer_FLOPs/1e9} GFLOPs")


