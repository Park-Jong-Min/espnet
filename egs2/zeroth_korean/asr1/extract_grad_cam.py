from zeroth_data_loader import zeroth_dataset
from zeroth_model_loader import load_model
from hook_related_fn import *
from model_dict import model_dict
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import argparse
import os
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def save_grad_image(image_list, target_list, audio_num, n_targets, PATH, mode, layer_num, head_num):
    fig_saved_dir = PATH

    for tar in range(n_targets):
        fig = plt.figure(figsize=(layer_num, head_num))
        if mode == 'mean' or mode == 'mean_att':
            img = np.array(image_list)
            img = np.mean(img, axis=0)

        else:
            img = image_list[tar]

        plt.imshow(img)
        plt.axis('off')

        print(f'process {target_list[tar]} target images....')

        plt.savefig(fig_saved_dir + f'/head_score_audio_{audio_num}_target_{target_list[tar]}.png',
                    bbox_inches='tight',
                    dpi=100)

        if mode == 'mean' or mode == 'mean_att':
            np.save(fig_saved_dir + f'/mean.npy', img)

        plt.close()

def make_hook_module_list(net):
    module_list = []

    for name, module in net.named_modules():
        if 'output_layer' in name:
            module_list.append((module, name))
        
        elif 'ctc.ctc_lo' in name:
            module_list.append((module, name))

        elif 'linear_out' in name:
            module_list.append((module, name))

    return module_list

def make_grad_cam_img_list(model, ctc_out, ctc_loss, att_out, att_loss, hook_list, ctc_weight, encoder_layer_num, decoder_layer_num, head_num, d_model, mode):
    
    model.zero_grad()

    encoder_self_image = torch.zeros((encoder_layer_num, head_num))
    decoder_self_image = torch.zeros((decoder_layer_num, head_num))
    decoder_src_image = torch.zeros((decoder_layer_num, head_num))

    if mode == 'sentence' or mode == 'mean':

        ctc_out.backward(gradient=ctc_loss, retain_graph=True)
        att_out.backward(gradient=att_loss, retain_graph=True)

        for layer_idx in range(encoder_layer_num):
            hook = hook_list[f'encoder.encoders.{layer_idx}.self_attn.linear_out']

            grad_out_head_view = (hook.grad_input[0] + hook.grad_input[1]).view(1, -1, head_num, int(d_model/head_num))
            gap_grad = torch.mean(grad_out_head_view, dim=3).unsqueeze(3)

            feature = hook.input[0].view(1, -1, head_num, int(d_model/head_num))
            grad_cam = F.relu(torch.mul(feature, gap_grad)).mean(dim=1).mean(dim=2)

            encoder_self_image[layer_idx] = grad_cam.detach().clone()
        
        for layer_idx in range(decoder_layer_num):
            hook = hook_list[f'decoder.decoders.{layer_idx}.self_attn.linear_out']

            grad_out_head_cat = torch.cat(hook.grad_input, dim=1)
            feature_cat = torch.cat(hook.input, dim=1)

            grad_out_head_view = grad_out_head_cat.view(1, -1, head_num, int(d_model/head_num))
            gap_grad = torch.mean(grad_out_head_view, dim=3).unsqueeze(3)

            feature = feature_cat.view(1, -1, head_num, int(d_model/head_num))
            grad_cam = F.relu(torch.mul(feature, gap_grad)).mean(dim=1).mean(dim=2)

            decoder_self_image[layer_idx] = grad_cam.detach().clone()
        
        for layer_idx in range(decoder_layer_num):
            hook = hook_list[f'decoder.decoders.{layer_idx}.src_attn.linear_out']

            grad_out_head_cat = torch.cat(hook.grad_input, dim=1)
            feature_cat = torch.cat(hook.input, dim=1)

            grad_out_head_view = grad_out_head_cat.view(1, -1, head_num, int(d_model/head_num))
            gap_grad = torch.mean(grad_out_head_view, dim=3).unsqueeze(3)

            feature = feature_cat.view(1, -1, head_num, int(d_model/head_num))
            grad_cam = F.relu(torch.mul(feature, gap_grad)).mean(dim=1).mean(dim=2)

            decoder_src_image[layer_idx] = grad_cam.detach().clone()
    
    else:
        att_out.backward(gradient=att_loss, retain_graph=True)

        for layer_idx in range(encoder_layer_num):
            hook = hook_list[f'encoder.encoders.{layer_idx}.self_attn.linear_out']

            grad_out_head_view = (hook.grad_input[0]).view(1, -1, head_num, int(d_model/head_num))
            gap_grad = torch.mean(grad_out_head_view, dim=3).unsqueeze(3)

            feature = hook.input[0].view(1, -1, head_num, int(d_model/head_num))
            grad_cam = F.relu(torch.mul(feature, gap_grad)).mean(dim=1).mean(dim=2)

            encoder_self_image[layer_idx] = grad_cam.detach().clone()
        
        for layer_idx in range(decoder_layer_num):
            hook = hook_list[f'decoder.decoders.{layer_idx}.self_attn.linear_out']

            grad_out_head_cat = torch.cat(hook.grad_input, dim=1)
            feature_cat = torch.cat(hook.input, dim=1)

            grad_out_head_view = grad_out_head_cat.view(1, -1, head_num, int(d_model/head_num))
            gap_grad = torch.mean(grad_out_head_view, dim=3).unsqueeze(3)

            feature = feature_cat.view(1, -1, head_num, int(d_model/head_num))
            grad_cam = F.relu(torch.mul(feature, gap_grad)).mean(dim=1).mean(dim=2)

            decoder_self_image[layer_idx] = grad_cam.detach().clone()
        
        for layer_idx in range(decoder_layer_num):
            hook = hook_list[f'decoder.decoders.{layer_idx}.src_attn.linear_out']

            grad_out_head_cat = torch.cat(hook.grad_input, dim=1)
            feature_cat = torch.cat(hook.input, dim=1)

            grad_out_head_view = grad_out_head_cat.view(1, -1, head_num, int(d_model/head_num))
            gap_grad = torch.mean(grad_out_head_view, dim=3).unsqueeze(3)

            feature = feature_cat.view(1, -1, head_num, int(d_model/head_num))
            grad_cam = F.relu(torch.mul(feature, gap_grad)).mean(dim=1).mean(dim=2)

            decoder_src_image[layer_idx] = grad_cam.detach().clone()

    return encoder_self_image.detach().numpy(), decoder_self_image.detach().numpy(), decoder_src_image.detach().numpy()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Grad-CAM Argparse')

    parser.add_argument('--audio-number', type=int, default=0, 
                        help='audio number for extract sentence and word grad cam')

    parser.add_argument('--mode', type=str, choices=['word', 'sentence', 'mean', 'mean_att'], default='sentence', 
                        help='word/sentence/mean mode for grad-cam')
    parser.add_argument('--ctc-weight', type=float, default=0.3, 
                        help='-weight for ctc one-hot bacpropagation')

    args = parser.parse_args()

    asr_exp_name = 'asr_TF_FF512_NoLM'
    asr_config_file = asr_exp_name + '/config.yaml'
    asr_model_file = asr_exp_name + '/valid.acc.ave.pth'
    exp_dir = '/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/exp'

    enc_layer_num = model_dict[asr_exp_name][0]
    dec_layer_num = model_dict[asr_exp_name][1]
    head_num = model_dict[asr_exp_name][2]
    d_model = model_dict[asr_exp_name][3]

    # parameter setting

    mode = args.mode

    if mode == 'mean' or mode == 'mean_att':
        enc_self_dir = f'/feature_images/{mode}/enc_self/audio_{asr_exp_name}'
        dec_self_dir = f'/feature_images/{mode}/dec_self/audio_{asr_exp_name}'
        dec_src_dir = f'/feature_images/{mode}/dec_src/audio_{asr_exp_name}'
    else:
        audio_num = args.audio_number
        enc_self_dir = f'/feature_images/{mode}/enc_self/{asr_exp_name}/audio_{audio_num}'
        dec_self_dir = f'/feature_images/{mode}/dec_self/{asr_exp_name}/audio_{audio_num}'
        dec_src_dir = f'/feature_images/{mode}/dec_src/{asr_exp_name}/audio_{audio_num}'


    ctc_weight = args.ctc_weight
    att_weight = 1 - ctc_weight

    createFolder(exp_dir + enc_self_dir)
    createFolder(exp_dir + dec_self_dir)
    createFolder(exp_dir + dec_src_dir)

    dataset = zeroth_dataset()
    speech2text = load_model(asr_config_file=asr_config_file, 
                            asr_model_file=asr_model_file, 
                            prune_ratio=0, 
                            prune_mode='no_prune',
                            device='cpu')

    module_list = make_hook_module_list(speech2text.asr_model)

    hookF = {name: GetHook(module) for (module, name) in module_list}

    enc_self_img_list = []
    dec_self_img_list = []
    dec_src_img_list = []
    word_num_list = []

    if mode == 'sentence':
        speech = dataset[audio_num][1]['speech']
        text = dataset[audio_num][1]['text']

        if isinstance(speech, np.ndarray):
            speech = torch.tensor(speech)
        
        out = speech2text(speech)

        decoder_out = torch.empty((len(hookF['decoder.output_layer'].output), 5000))

        for i in range(len(hookF['decoder.output_layer'].output)):
            decoder_out[i] = hookF['decoder.output_layer'].output[i]
        
        ctc_out = hookF['ctc.ctc_lo'].output[0]

        ctc_out.unsqueeze_(0)
        decoder_out.unsqueeze_(0)

        ctc_argmax = ctc_out.argmax(2)
        att_argmax = decoder_out.argmax(2)

        one_hot_ctc = torch.zeros_like(ctc_out)
        one_hot_ctc.scatter_(2, ctc_argmax.unsqueeze(2), ctc_weight)

        one_hot_att = torch.zeros_like(decoder_out)
        one_hot_att.scatter_(2, att_argmax.unsqueeze(2), att_weight)

        enc_self_img, dec_self_img, dec_src_img = make_grad_cam_img_list(model=speech2text.asr_model, 
                                                                        ctc_out=ctc_out, 
                                                                        ctc_loss=one_hot_ctc, 
                                                                        att_out=decoder_out, 
                                                                        att_loss=one_hot_att, 
                                                                        hook_list=hookF, 
                                                                        ctc_weight=ctc_weight,
                                                                        encoder_layer_num=enc_layer_num, 
                                                                        decoder_layer_num=dec_layer_num, 
                                                                        head_num=head_num, 
                                                                        d_model=d_model,
                                                                        mode=mode)
        
        enc_self_img_list.append(enc_self_img)
        dec_self_img_list.append(dec_self_img)
        dec_src_img_list.append(dec_src_img)
        word_num_list.append('audio'+str(audio_num))

        save_grad_image(image_list=enc_self_img_list, target_list=word_num_list, audio_num=audio_num,
                        n_targets=1, 
                        PATH=exp_dir + enc_self_dir,
                        layer_num=enc_layer_num,
                        head_num=head_num,
                        mode=mode)
        
        save_grad_image(image_list=dec_self_img_list, target_list=word_num_list, audio_num=audio_num,
                        n_targets=1, 
                        PATH=exp_dir + dec_self_dir,
                        layer_num=dec_layer_num,
                        head_num=head_num,
                        mode=mode)
        
        save_grad_image(image_list=dec_src_img_list, target_list=word_num_list, audio_num=audio_num,
                        n_targets=1, 
                        PATH=exp_dir + dec_src_dir,
                        layer_num=dec_layer_num,
                        head_num=head_num,
                        mode=mode)
    elif mode == 'word':

        space = False
        n_targets = 0

        speech = dataset[audio_num][1]['speech']

        out = speech2text(speech)

        decoder_out = torch.empty((len(hookF['decoder.output_layer'].output), 5000))

        for i in range(len(hookF['decoder.output_layer'].output)):
            decoder_out[i] = hookF['decoder.output_layer'].output[i]
        
        decoder_out.unsqueeze_(0)

        att_argmax = decoder_out.argmax(2)

        for tar in range(decoder_out.size(1)):
            if att_argmax[0, tar] == 0:
                if space == False:
                    continue
                else:
                    n_targets += 1
                    one_hot = torch.zeros_like(decoder_out)
                    one_hot[0, tar, att_argmax[0,tar].item()] = 1

                    enc_self_img, dec_self_img, dec_src_img = make_grad_cam_img_list(model=speech2text.asr_model, 
                                                                        ctc_out=None, 
                                                                        ctc_loss=None, 
                                                                        att_out=decoder_out, 
                                                                        att_loss=one_hot, 
                                                                        hook_list=hookF, 
                                                                        ctc_weight=ctc_weight,
                                                                        encoder_layer_num=enc_layer_num, 
                                                                        decoder_layer_num=dec_layer_num, 
                                                                        head_num=head_num, 
                                                                        d_model=d_model,
                                                                        mode=mode)

                    enc_self_img_list.append(enc_self_img)
                    dec_self_img_list.append(dec_self_img)
                    dec_src_img_list.append(dec_src_img)
                    word_num_list.append(f'{tar}_{att_argmax[0,tar]}')

                    for name in hookF:
                        hookF[name].grad_list_reset()
            
            else:
                n_targets += 1
                one_hot = torch.zeros_like(decoder_out)
                one_hot[0, tar, att_argmax[0,tar].item()] = 1

                enc_self_img, dec_self_img, dec_src_img = make_grad_cam_img_list(model=speech2text.asr_model, 
                                                                    ctc_out=None, 
                                                                    ctc_loss=None, 
                                                                    att_out=decoder_out, 
                                                                    att_loss=one_hot, 
                                                                    hook_list=hookF, 
                                                                    ctc_weight=ctc_weight,
                                                                    encoder_layer_num=enc_layer_num, 
                                                                    decoder_layer_num=dec_layer_num, 
                                                                    head_num=head_num, 
                                                                    d_model=d_model,
                                                                    mode=mode)

                enc_self_img_list.append(enc_self_img)
                dec_self_img_list.append(dec_self_img)
                dec_src_img_list.append(dec_src_img)
                word_num_list.append(f'{tar}_{att_argmax[0,tar]}')

                for name in hookF:
                    hookF[name].grad_list_reset()

        save_grad_image(image_list=enc_self_img_list, target_list=word_num_list, audio_num=audio_num,
                        n_targets=len(word_num_list), 
                        PATH=exp_dir + enc_self_dir,
                        layer_num=enc_layer_num,
                        head_num=head_num,
                        mode=mode)
        
        save_grad_image(image_list=dec_self_img_list, target_list=word_num_list, audio_num=audio_num,
                        n_targets=len(word_num_list), 
                        PATH=exp_dir + dec_self_dir,
                        layer_num=dec_layer_num,
                        head_num=head_num,
                        mode=mode)
        
        save_grad_image(image_list=dec_src_img_list, target_list=word_num_list, audio_num=audio_num,
                        n_targets=len(word_num_list), 
                        PATH=exp_dir + dec_src_dir,
                        layer_num=dec_layer_num,
                        head_num=head_num,
                        mode=mode)
        
    elif mode == 'mean' or mode == 'mean_att':
        data_size = 220
        # data_size = 3

        for i in tqdm(range(data_size)):
            speech = dataset[i][1]['speech']

            if isinstance(speech, np.ndarray):
                speech = torch.tensor(speech)
            
            out = speech2text(speech)

            decoder_out = torch.empty((len(hookF['decoder.output_layer'].output), 5000))

            for i in range(len(hookF['decoder.output_layer'].output)):
                decoder_out[i] = hookF['decoder.output_layer'].output[i]
            
            ctc_out = hookF['ctc.ctc_lo'].output[0]

            ctc_out.unsqueeze_(0)
            decoder_out.unsqueeze_(0)

            ctc_argmax = ctc_out.argmax(2)
            att_argmax = decoder_out.argmax(2)

            one_hot_ctc = torch.zeros_like(ctc_out)
            one_hot_ctc.scatter_(2, ctc_argmax.unsqueeze(2), ctc_weight)

            one_hot_att = torch.zeros_like(decoder_out)
            one_hot_att.scatter_(2, att_argmax.unsqueeze(2), att_weight)

            enc_self_img, dec_self_img, dec_src_img = make_grad_cam_img_list(model=speech2text.asr_model, 
                                                                            ctc_out=ctc_out, 
                                                                            ctc_loss=one_hot_ctc, 
                                                                            att_out=decoder_out, 
                                                                            att_loss=one_hot_att, 
                                                                            hook_list=hookF, 
                                                                            ctc_weight=ctc_weight,
                                                                            encoder_layer_num=enc_layer_num, 
                                                                            decoder_layer_num=dec_layer_num, 
                                                                            head_num=head_num, 
                                                                            d_model=d_model,
                                                                            mode=mode)
            
            enc_self_img_list.append(enc_self_img)
            dec_self_img_list.append(dec_self_img)
            dec_src_img_list.append(dec_src_img)
            word_num_list.append(mode)

            for name in hookF:
                hookF[name].list_reset()

        save_grad_image(image_list=enc_self_img_list, target_list=word_num_list, audio_num='mean',
                        n_targets=1, 
                        PATH=exp_dir + enc_self_dir,
                        layer_num=enc_layer_num,
                        head_num=head_num,
                        mode=mode)
        
        save_grad_image(image_list=dec_self_img_list, target_list=word_num_list, audio_num='mean',
                        n_targets=1, 
                        PATH=exp_dir + dec_self_dir,
                        layer_num=dec_layer_num,
                        head_num=head_num,
                        mode=mode)
        
        save_grad_image(image_list=dec_src_img_list, target_list=word_num_list, audio_num='mean',
                        n_targets=1, 
                        PATH=exp_dir + dec_src_dir,
                        layer_num=dec_layer_num,
                        head_num=head_num,
                        mode=mode)
