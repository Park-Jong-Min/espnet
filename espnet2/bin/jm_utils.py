import numpy as np
import torch
import torch.nn.utils.prune as prune

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

def global_head_mask(model, prune_ratio):

    n_heads = 8
    d_model = 512
    d_k = int(d_model / n_heads)
    weight_dict = {}
    mask_dict = {}

    for name, param in model.named_parameters():
        if 'linear_q.weight' in name or 'linear_k.weight' in name or 'linear_v.weight' in name:
            mask_dict[name] = torch.ones_like(param)
            for head in range(n_heads):
                weight_dict[name + f'_{head}'] = param.T[:, d_k*(head):d_k*(head+1)].abs().mean()
    
    weight_dict = sorted(weight_dict, key=lambda k : weight_dict[k])

    for i in range(int(len(weight_dict)*prune_ratio)):
        mask = torch.ones((d_model, d_model))
        head = int(weight_dict[i][-1])

        mask.T[:, d_k*(head):d_k*(head+1)] = 0
        mask_dict[weight_dict[i][:-9]] = mask

    return mask_dict


def global_pruning(model, prune_ratio, re_param=True):

    mask_dict = global_head_mask(model, prune_ratio)

    for name, module in model.named_modules():
        if name in mask_dict:
            prune.custom_from_mask(module, name='weight', mask=mask_dict[name])
            if re_param == True:
                prune.remove(module, 'weight')
    