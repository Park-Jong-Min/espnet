import torch

class GetHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.pre_hook = module.register_forward_pre_hook(self.pre_hook_fn)
        self.input = []
        self.output = []
        self.grad_input = []
        self.grad_output = []

    def hook_fn(self, module, input, output):
        self.input.append(input[0])
        self.output.append(output)

        return output
    
    def pre_hook_fn(self, module, input):
        input[0].register_hook(lambda grad: self.grad_input.append(grad.detach().clone()))
        
    def close(self):
        self.hook.remove()

    def list_reset(self):
        self.input = []
        self.output = []
        self.grad_input = []
        self.grad_output = []
    
    def grad_list_reset(self):
        self.grad_input = []
        self.grad_output = []

class OutHook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.input = []
        self.output = []
    
    def hook_fn(self, module, input, output):
        self.input.append(input)
        self.output.append(output)
    
    def reset(self):
        self.input = []
        self.output = []
        
def apply_GetHook(model, module_list):
    # module_type : 'encoder', 'decoder'
    # attn_type : 'self_attn', 'src_attn'

    hook_list = []

    for name, param in model.named_modules():
        if name in module_list:
            hook_list.append(GetHook(module=param))
    
    return hook_list

def apply_OutHook(model, module_list):
    # module_type : 'encoder', 'decoder'
    # attn_type : 'self_attn', 'src_attn'

    hook_list = []

    for name, param in model.named_modules():
        if name in module_list:
            hook_list.append(OutHook(module=param))
    
    return hook_list
