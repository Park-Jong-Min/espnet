from zeroth_data_loader import *
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.bin.asr_inference import Speech2Text
from espnet2.tasks.asr import ASRTask
from espnet_model_zoo.downloader import ModelDownloader

class Hook_In_Out():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.input = []
        self.output = []

    def hook_fn(self, module, input, output):
        self.input.append(input)
        self.output.append(output)
        
    def close(self):
        self.hook.remove()

def apply_hook(net):
    for name, module in net.named_modules():
        if f'decoder.output_layer' == name:
            hook = Hook_In_Out(module=module)
            return hook

def load_model(asr_config_file, asr_model_file, prune_ratio, prune_mode, device='cpu'):

    exp_path = '/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/exp/'
    bpe_model = '/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/data/kr_token_list/bpe_unigram5000/bpe.model'
    d = ModelDownloader()

    speech2text = Speech2Text(
    # **d.download_and_unpack('Hoon Chung/zeroth_korean_asr_train_asr_transformer5_raw_bpe_valid.acc.ave'),
    # Decoding parameters are not included in the model file
    asr_train_config=exp_path + asr_config_file,
    asr_model_file=exp_path + asr_model_file,
    token_type='bpe',
    bpemodel=bpe_model,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=1,
    ctc_weight=0.3,
    lm_weight=0.0,
    penalty=0.0,
    nbest=1,
    device=device,
    prune_ratio=prune_ratio,
    prune_mode=prune_mode,
    )

    return speech2text

if __name__ == "__main__":
    asr_config_file = 'asr_train_asr_transformer5_raw_kr_bpe5000/config.yaml'
    asr_model_file = 'asr_train_asr_transformer5_raw_kr_bpe5000/98epoch.pth'
    model = load_model(asr_config_file, asr_model_file)