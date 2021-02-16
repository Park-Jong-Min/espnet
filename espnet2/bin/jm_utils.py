import numpy as np
import torch
import torch.nn.utils.prune as prune
from model_dict import model_dict

def make_prune_list(np_file_path, prune_ratio, mode='lower'):
    # prune ratio is 0~1 float
    # mode = 'upper' or 'lower' -> upper is delete high value head and lower does opposite role
    mean_img = np.load(np_file_path)
    # numpy file has (18, 8) shape

    total_head_num = int(18 * 8 * prune_ratio)

    prune_list_np = np.zeros((18, 8))

    if mode == 'lower':
        sorted_index = np.dstack(np.unravel_index(np.argsort(mean_img.ravel()), (18, 8))) # ascending order

        for i in range(total_head_num):
            prune_list_np[sorted_index[0][i][0], sorted_index[0][i][1]] = 1
        
        delete_list = []
        layer_list = []

        for x in range(18):
            temp_list = []
            for y in range(8):
                if prune_list_np[x, y] == 1:
                    temp_list.append(y)
            
            if temp_list:
                delete_list.append(temp_list)
                layer_list.append(x)

        return layer_list, delete_list

    else:
        sorted_index = np.dstack(np.unravel_index(np.argsort(-mean_img.ravel()), (18, 8))) # ascending order
        
        for i in range(total_head_num):
            prune_list_np[sorted_index[0][i][0], sorted_index[0][i][1]] = 1
        
        delete_list = []
        layer_list = []

        for x in range(18):
            temp_list = []
            for y in range(8):
                if prune_list_np[x, y] == 1:
                    temp_list.append(y)
            
            if temp_list:
                delete_list.append(temp_list)
                layer_list.append(x)

        return layer_list, delete_list

class Hook():
    def __init__(self, module, layer_idx, survive_head_idx, logging, module_type, attn_type):
        self.hook = module.register_forward_pre_hook(self.hook_fn)
        self.layer_idx = layer_idx
        self.survive_head_idx = survive_head_idx
        self.name = str(self.layer_idx) + '_' + str(self.survive_head_idx)

        logging.info(f"Setting {self.name} hook.")

    def hook_fn(self, module, input):
        return input[0], input[1], input[2], input[3], self.survive_head_idx
        
    def close(self):
        self.hook.remove()

class PruneHook():
    def __init__(self, module, layer_idx, delete_head_list, logging, module_type, attn_type):
        self.hook = module.register_forward_pre_hook(self.hook_fn)
        self.layer_idx = layer_idx
        self.delete_head_list = delete_head_list
        self.name = str(self.layer_idx) + '_' + str(self.delete_head_list)

        logging.info(f"Setting {self.name} hook.")

    def hook_fn(self, module, input):

        return input[0], input[1], input[2], input[3], torch.tensor(self.delete_head_list)
        
    def close(self):
        self.hook.remove()
    
def apply_hook(model, layer_idx, head_idx, module_type, attn_type, logging):
    # module_type : 'encoder', 'decoder'
    # attn_type : 'self_attn', 'src_attn'

    for name, param in model.named_modules():
        if f'{module_type}.{module_type}s.{layer_idx}.{attn_type}' == name:
            Hook(module=param,
                layer_idx=layer_idx, survive_head_idx=head_idx,
                logging=logging,
                module_type=module_type, attn_type=attn_type)

def apply_prunehook(model, NP_PATH, prune_ratio, module_type, attn_type, logging):
    # module_type : 'encoder', 'decoder'
    # attn_type : 'self_attn', 'src_attn'

    layer_list, head_list = make_prune_list(np_file_path=NP_PATH, prune_ratio=prune_ratio, mode='lower')

    for name, param in model.named_modules():
        for i, layer_idx in enumerate(layer_list):
            if f'{module_type}.{module_type}s.{layer_idx}.{attn_type}' == name:
                PruneHook(module=param,
                        layer_idx=layer_idx, delete_head_list=head_list[i],
                        logging=logging,
                        module_type=module_type, attn_type=attn_type)

class GetHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.input = None
        self.output = None

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output
        
    def close(self):
        self.hook.remove()

def apply_output_hook(model):

    for name, module in model.named_modules():
        if 'ctc.ctc_lo' == name:
            ctc_hook = GetHook(module)
        
        elif 'criterion_att.criterion':
            att_hook = GetHook(module)
    
    return ctc_hook, att_hook

def global_head_mask(model, prune_ratio, prune_mode, exp_dir, exp_name):

    enc_layer_num = model_dict[exp_name][0]
    dec_layer_num = model_dict[exp_name][1]
    n_heads = model_dict[exp_name][2]
    d_model = model_dict[exp_name][3]

    d_k = int(d_model / n_heads)

    enc_self_weight_dict = {}
    dec_self_weight_dict = {}
    dec_src_weight_dict = {}

    mask_dict = {}

    for name, param in model.named_parameters():
        if 'linear_q.weight' in name and 'encoder' in name:
            mask_dict[name] = torch.ones_like(param)
            for head in range(n_heads):
                enc_self_weight_dict[name[:-16] + f'_{head}'] = param.T[:, d_k*(head):d_k*(head+1)].abs().mean()

        if ('linear_k.weight' in name or 'linear_v.weight' in name) and ('encoder' in name):
            mask_dict[name] = torch.ones_like(param)
            for head in range(n_heads):
                enc_self_weight_dict[name[:-16] + f'_{head}'] += param.T[:, d_k*(head):d_k*(head+1)].abs().mean()
    
    for name, param in model.named_parameters():
        if 'linear_q.weight' in name and 'decoder' in name and 'self_attn' in name:
            mask_dict[name] = torch.ones_like(param)
            for head in range(n_heads):
                dec_self_weight_dict[name[:-16] + f'_{head}'] = param.T[:, d_k*(head):d_k*(head+1)].abs().mean()

        if ('linear_k.weight' in name or 'linear_v.weight' in name) and ('decoder' in name and 'self_attn' in name):
            mask_dict[name] = torch.ones_like(param)
            for head in range(n_heads):
                dec_self_weight_dict[name[:-16] + f'_{head}'] += param.T[:, d_k*(head):d_k*(head+1)].abs().mean()
    
    for name, param in model.named_parameters():
        if 'linear_q.weight' in name and 'decoder' in name and 'src_attn' in name:
            mask_dict[name] = torch.ones_like(param)
            for head in range(n_heads):
                dec_src_weight_dict[name[:-16] + f'_{head}'] = param.T[:, d_k*(head):d_k*(head+1)].abs().mean()

        if ('linear_k.weight' in name or 'linear_v.weight' in name) and ('decoder' in name and 'src_attn' in name):
            mask_dict[name] = torch.ones_like(param)
            for head in range(n_heads):
                dec_src_weight_dict[name[:-16] + f'_{head}'] += param.T[:, d_k*(head):d_k*(head+1)].abs().mean()
    
    if prune_mode == 'global_enc_self':
        weight_dict = enc_self_weight_dict
        weight_dict = sorted(weight_dict, key=lambda k : weight_dict[k])
    
        for i in range(int(len(weight_dict)*prune_ratio)):
            mask = torch.ones((d_model, d_model))

            mask_dict[weight_dict[i][:-2] + '.linear_q'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_k'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_v'] = mask.detach().clone()

        for i in range(int(len(weight_dict)*prune_ratio)):
            head = int(weight_dict[i][-1])

            mask_dict[weight_dict[i][:-2] + '.linear_q'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_k'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_v'].T[:, d_k*(head):d_k*(head+1)] = 0 

    elif prune_mode == 'global_dec_self':
        weight_dict = dec_self_weight_dict
        weight_dict = sorted(weight_dict, key=lambda k : weight_dict[k])
        for i in range(int(len(weight_dict)*prune_ratio)):
            mask = torch.ones((d_model, d_model))

            mask_dict[weight_dict[i][:-2] + '.linear_q'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_k'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_v'] = mask.detach().clone()

        for i in range(int(len(weight_dict)*prune_ratio)):
            head = int(weight_dict[i][-1])

            mask_dict[weight_dict[i][:-2] + '.linear_q'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_k'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_v'].T[:, d_k*(head):d_k*(head+1)] = 0 
    elif prune_mode == 'global_dec_src':
        weight_dict = dec_src_weight_dict
        weight_dict = sorted(weight_dict, key=lambda k : weight_dict[k])
        for i in range(int(len(weight_dict)*prune_ratio)):
            mask = torch.ones((d_model, d_model))

            mask_dict[weight_dict[i][:-2] + '.linear_q'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_k'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_v'] = mask.detach().clone()

        for i in range(int(len(weight_dict)*prune_ratio)):
            head = int(weight_dict[i][-1])

            mask_dict[weight_dict[i][:-2] + '.linear_q'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_k'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_v'].T[:, d_k*(head):d_k*(head+1)] = 0 
    
    elif prune_mode == 'global':
        weight_dict_list = [enc_self_weight_dict, dec_self_weight_dict, dec_src_weight_dict]
        for i in range(3):
            weight_dict = weight_dict_list[i]
            weight_dict = sorted(weight_dict, key=lambda k : weight_dict[k])

            for i in range(int(len(weight_dict)*prune_ratio)):
                mask = torch.ones((d_model, d_model))

                mask_dict[weight_dict[i][:-2] + '.linear_q'] = mask.detach().clone()
                mask_dict[weight_dict[i][:-2] + '.linear_k'] = mask.detach().clone()
                mask_dict[weight_dict[i][:-2] + '.linear_v'] = mask.detach().clone()

            for i in range(int(len(weight_dict)*prune_ratio)):
                head = int(weight_dict[i][-1])

                mask_dict[weight_dict[i][:-2] + '.linear_q'].T[:, d_k*(head):d_k*(head+1)] = 0
                mask_dict[weight_dict[i][:-2] + '.linear_k'].T[:, d_k*(head):d_k*(head+1)] = 0
                mask_dict[weight_dict[i][:-2] + '.linear_v'].T[:, d_k*(head):d_k*(head+1)] = 0 
        
    return mask_dict

def grad_head_mask(model, prune_ratio, prune_mode, exp_dir, exp_name):

    enc_layer_num = model_dict[exp_name][0]
    dec_layer_num = model_dict[exp_name][1]
    n_heads = model_dict[exp_name][2]
    d_model = model_dict[exp_name][3]
    d_k = int(d_model / n_heads)

    enc_self_weight_dict = {}
    dec_self_weight_dict = {}
    dec_src_weight_dict = {}

    mask_dict = {}

    enc_self_dir = f'/feature_images/mean_att/enc_self/audio_{exp_name}/mean.npy'
    dec_self_dir = f'/feature_images/mean_att/dec_self/audio_{exp_name}/mean.npy'
    dec_src_dir = f'/feature_images/mean_att/dec_src/audio_{exp_name}/mean.npy'

    enc_self_mean = np.load(exp_dir + enc_self_dir)
    dec_self_mean = np.load(exp_dir + dec_self_dir)
    dec_src_mean = np.load(exp_dir + dec_src_dir)

    for layer in range(enc_layer_num):
        for head in range(n_heads):
            enc_self_weight_dict[f'encoder.encoders.{layer}.self_attn_{head}'] = enc_self_mean[layer, head] 

    for layer in range(dec_layer_num):
        for head in range(n_heads):
            dec_self_weight_dict[f'decoder.decoders.{layer}.self_attn_{head}'] = dec_self_mean[layer, head]
            dec_src_weight_dict[f'decoder.decoders.{layer}.src_attn_{head}'] = dec_src_mean[layer, head]
    
    if prune_mode == 'grad_enc_self':
        weight_dict = enc_self_weight_dict
        weight_dict = sorted(weight_dict, key=lambda k : weight_dict[k])
    
        for i in range(int(len(weight_dict)*prune_ratio)):
            mask = torch.ones((d_model, d_model))

            mask_dict[weight_dict[i][:-2] + '.linear_q'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_k'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_v'] = mask.detach().clone()

        for i in range(int(len(weight_dict)*prune_ratio)):
            head = int(weight_dict[i][-1])

            mask_dict[weight_dict[i][:-2] + '.linear_q'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_k'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_v'].T[:, d_k*(head):d_k*(head+1)] = 0 

    elif prune_mode == 'grad_dec_self':
        weight_dict = dec_self_weight_dict
        weight_dict = sorted(weight_dict, key=lambda k : weight_dict[k])
        for i in range(int(len(weight_dict)*prune_ratio)):
            mask = torch.ones((d_model, d_model))

            mask_dict[weight_dict[i][:-2] + '.linear_q'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_k'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_v'] = mask.detach().clone()

        for i in range(int(len(weight_dict)*prune_ratio)):
            head = int(weight_dict[i][-1])

            mask_dict[weight_dict[i][:-2] + '.linear_q'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_k'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_v'].T[:, d_k*(head):d_k*(head+1)] = 0 
    elif prune_mode == 'grad_dec_src':
        weight_dict = dec_src_weight_dict
        weight_dict = sorted(weight_dict, key=lambda k : weight_dict[k])
        for i in range(int(len(weight_dict)*prune_ratio)):
            mask = torch.ones((d_model, d_model))

            mask_dict[weight_dict[i][:-2] + '.linear_q'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_k'] = mask.detach().clone()
            mask_dict[weight_dict[i][:-2] + '.linear_v'] = mask.detach().clone()

        for i in range(int(len(weight_dict)*prune_ratio)):
            head = int(weight_dict[i][-1])

            mask_dict[weight_dict[i][:-2] + '.linear_q'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_k'].T[:, d_k*(head):d_k*(head+1)] = 0
            mask_dict[weight_dict[i][:-2] + '.linear_v'].T[:, d_k*(head):d_k*(head+1)] = 0 
    
    elif prune_mode == 'grad':
        weight_dict_list = [enc_self_weight_dict, dec_self_weight_dict, dec_src_weight_dict]
        for i in range(3):
            weight_dict = weight_dict_list[i]
            weight_dict = sorted(weight_dict, key=lambda k : weight_dict[k])

            for i in range(int(len(weight_dict)*prune_ratio)):
                mask = torch.ones((d_model, d_model))

                mask_dict[weight_dict[i][:-2] + '.linear_q'] = mask.detach().clone()
                mask_dict[weight_dict[i][:-2] + '.linear_k'] = mask.detach().clone()
                mask_dict[weight_dict[i][:-2] + '.linear_v'] = mask.detach().clone()

            for i in range(int(len(weight_dict)*prune_ratio)):
                head = int(weight_dict[i][-1])

                mask_dict[weight_dict[i][:-2] + '.linear_q'].T[:, d_k*(head):d_k*(head+1)] = 0
                mask_dict[weight_dict[i][:-2] + '.linear_k'].T[:, d_k*(head):d_k*(head+1)] = 0
                mask_dict[weight_dict[i][:-2] + '.linear_v'].T[:, d_k*(head):d_k*(head+1)] = 0 
        
    return mask_dict


def pruning(model, prune_ratio, prune_mode, exp_name, re_param=True, device='cpu'):

    exp_dir = '/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/exp'

    if prune_mode == 'global_enc_self' or prune_mode == 'global_dec_self' or prune_mode == 'global_dec_src' or prune_mode == 'global':
        mask_dict = global_head_mask(model, prune_ratio, prune_mode, exp_dir, exp_name)

        for name, module in model.named_modules():
            if name in mask_dict:
                prune.custom_from_mask(module, name='weight', mask=mask_dict[name].to(device))
                if re_param == True:
                    prune.remove(module, 'weight')
    
    elif prune_mode == 'grad_enc_self' or prune_mode == 'grad_dec_self' or prune_mode == 'grad_dec_src' or prune_mode == 'grad':
        mask_dict = grad_head_mask(model, prune_ratio, prune_mode, exp_dir, exp_name)

        for name, module in model.named_modules():
            if name in mask_dict:
                prune.custom_from_mask(module, name='weight', mask=mask_dict[name].to(device))
                if re_param == True:
                    prune.remove(module, 'weight')
    
    elif prune_mode == 'no_prune':
        pass
    
    else:
        print(prune_mode)
        print(prune_ratio)
        print("PRUNE MODE ERROR!! NO NAME EXIST")

def prune_remove(model):
    mask_module_list = []
    mask_dict = {}

    for name, param in model.named_buffers():
        if 'attn' in name:
            mask_module_list.append(name[:-12])
            mask_dict[name[:-12]] = param
    
    for name, module in model.named_modules():
        if name in mask_module_list:
            prune.remove(module, 'weight')

    return mask_dict
    
def prune_remask(model, mask_dict):
    for name, module in model.named_modules():
        if name in mask_dict:
            prune.custom_from_mask(module, name='weight', mask=mask_dict[name])