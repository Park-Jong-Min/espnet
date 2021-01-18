import sentencepiece as spm
from espnet2.tasks.abs_task import AbsTask
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.bin.asr_inference import Speech2Text
from espnet2.tasks.asr import ASRTask
from espnet2.bin.jm_utils import *
from espnet2.train.dataset import ESPnetDataset


wav_scp = '/home/jmpark/home_data_jmpark/espnet/egs2/jm_ref/asr1/dump/raw/dev_clean/wav.scp'
text = '/home/jmpark/home_data_jmpark/espnet/egs2/jm_ref/asr1/dump/raw/dev_clean/text'
token_int = '/home/jmpark/home_data_jmpark/espnet/egs2/jm_ref/asr1/dump/raw/dev_clean/text'
bpe_model = '/home/jmpark/home_data_jmpark/espnet/egs2/jm_ref/asr1/data/en_token_list/bpe_unigram5000/bpe.model'

dataset = ESPnetDataset([(wav_scp, 'speech', 'sound')])

sp = spm.SentencePieceProcessor()
sp.Load(bpe_model)

print(dataset[0])

f = open(text, 'r')

for id in dataset:
    sentence = f.readline()[17:]
    print(sentence)

f.close()