import numpy as np
import torch

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