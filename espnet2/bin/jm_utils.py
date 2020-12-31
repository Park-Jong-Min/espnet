
class delete_head_hook():
    def __init__(self, layer_idx, survive_head_idx, name='encoder_attn_'):
        self.layer_idx = layer_idx
        self.survive_head_idx = survive_head_idx
        self.name = name + str(self.layer_idx) + '_' + str(self.survive_head_idx)

    def hook(self, model, input, output):
        input[4][0] = self.survive_head_idx
    
    def close(self):
        self.hook.remove()

class Hook():
    def __init__(self, module, layer_idx, survive_head_idx, logging, module_type, attn_type, backward=False):
        if backward==False:
            self.hook = module.register_forward_pre_hook(self.hook_fn)
        else:
            self.hook = module.register_backward_hook(self.hook_fn)

        self.layer_idx = layer_idx
        self.survive_head_idx = survive_head_idx
        self.name = str(self.layer_idx) + '_' + str(self.survive_head_idx)

        logging.info(f"Setting {self.name} hook.")

    def hook_fn(self, module, input):
        query, key, value, mask, head_idx = input
        head_idx[0] = self.survive_head_idx
        return query, key, value, mask, head_idx
        
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