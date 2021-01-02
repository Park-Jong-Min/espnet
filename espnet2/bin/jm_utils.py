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
    
def apply_hook(model, layer_idx, head_idx, module_type, attn_type, logging):
    # module_type : 'encoder', 'decoder'
    # attn_type : 'self_attn', 'src_attn'

    for name, param in model.named_modules():
        if f'{module_type}.{module_type}s.{layer_idx}.{attn_type}' == name:
            Hook(module=param,
                layer_idx=layer_idx, survive_head_idx=head_idx,
                logging=logging,
                module_type=module_type, attn_type=attn_type)