from zeroth_data_loader import zeroth_dataset
from zeroth_model_loader import load_model
import numpy as np
import torch





if __name__ == "__main__":

    audio_num = 0

    dataset = zeroth_dataset()
    speech2text = load_model()
    
    speech = dataset[audio_num][1]['speech']

    # if isinstance(speech, np.ndarray):
    #     speech = torch.tensor(speech)

    # speech = speech.unsqueeze(0).to(getattr(torch, "float32"))
    # speech_lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))


    # text = dataset[audio_num][1]['token_idx']

    # if isinstance(text, np.ndarray):
    #     text = torch.tensor(text)

    # text = text.unsqueeze(0).to(getattr(torch, "int64"))
    # text_lengths = text.new_full([1], dtype=torch.long, fill_value=text.size(1))
    

    # batch = {"speech": speech, "speech_lengths": speech_lengths, "text": text, "text_lengths":text_lengths}

    # out = speech2text.asr_model(**batch)

    out = speech2text(speech)
    print(out)
    # print(speech)
    # print(speech.shape)