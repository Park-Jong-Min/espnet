from espnet2.bin.jm_utils import *
from espnet2.train.dataset import ESPnetDataset
from espnet2.train.preprocessor import CommonPreprocessor

def zeroth_dataset():
    wav_scp = '/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/dump/raw/train_dev/wav.scp'
    text = '/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/dump/raw/train_dev/text'
    bpe_model = '/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/data/kr_token_list/bpe_unigram5000/bpe.model'
    token_list = '/home/jmpark/home_data_jmpark/espnet/egs2/zeroth_korean/asr1/data/kr_token_list/bpe_unigram5000/tokens.txt'

    retval = CommonPreprocessor(
        train=True,
        token_type='bpe',
        token_list=token_list,
        bpemodel=bpe_model,
        space_symbol='_',
        non_linguistic_symbols=None,
        text_cleaner=None,
        g2p_type=None
    )

    dataset = ESPnetDataset([(wav_scp, 'speech', 'sound'),
                            (text, 'text', 'text')],
                            preprocess=retval
    )

    # dataset = ESPnetDataset([(wav_scp, 'speech', 'sound')]
    # )
    
    return dataset

if __name__ == "__main__":
    zeroth_dataset()