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

def load_model():
    bpe_model = '/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/data/kr_token_list/bpe_unigram5000/bpe.model'
    d = ModelDownloader()

    speech2text = Speech2Text(
    # **d.download_and_unpack('Hoon Chung/zeroth_korean_asr_train_asr_transformer5_raw_bpe_valid.acc.ave'),
    # Decoding parameters are not included in the model file
    asr_train_config='/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/exp/asr_train_asr_transformer5_raw_kr_bpe5000/config.yaml',
    asr_model_file='/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/exp/asr_train_asr_transformer5_raw_kr_bpe5000/98epoch.pth',
    token_type='bpe',
    bpemodel=bpe_model,
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=1,
    ctc_weight=0.3,
    lm_weight=0.0,
    penalty=0.0,
    nbest=1
    )

    att_hook = apply_hook(speech2text.asr_model)

    return speech2text, att_hook

if __name__ == "__main__":
    load_model()