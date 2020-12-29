import soundfile
import torch
import matplotlib.pyplot as plt
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

def save_image(image_list, name, n_layers, n_heads):
    rows = n_layers
    cols = n_heads
    fig_saved_dir = './exp/feature_images/'

    for j in range(rows):
        fig = plt.figure(figsize=(100,100))
        axes = []
        for i in range(cols):
            img = image_list[j*cols+i]
            axes.append(fig.add_subplot(1, cols, i+1))
            plt.imshow(img)
            plt.axis('off')

        print('process {0} layer images....'.format(j))
        plt.savefig(fig_saved_dir + 'audio' + str(audio_num) + '_' + name +'_layer{0}_attention.png'.format(j),
                    bbox_inches='tight',
                    dpi=100)

    # Use show when you want to show your attention image while extraction
    # plt.show() 

# encoder's attention extract function
saved_encoder_self_attn_images = []
saved_deocder_self_attn_images = []
def attn_encoder(self, input_tensor, output_tensor):
    cols = output_tensor[1].shape[1]

    for i in range(cols):
        img = output_tensor[1][0,i,:]
        saved_encoder_self_attn_images.append(img)

def attn_decoder(self, input_tensor, output_tensor):
    print("Not implemented")

# Set test wav for attention image extraction
TEST_DATA_PATH = "./data/dev_clean"
WAV_LIST_PATH = TEST_DATA_PATH + "/wav.scp"

file_name_list = []
audio_num = 1 # selelct one of the wav in file_name_list

with open(WAV_LIST_PATH, "r") as f:
    lines = f.readlines()
    for line in lines:
        num, name = line.split(' ')
        file_name_list.append(name[:-1])

speech, rate = soundfile.read(file_name_list[audio_num])

# Prepare model
d = ModelDownloader()

ASR_MODEL_PATH = "/home/jmpark/espnet/tools/anaconda/envs/espnet_1.7/lib/python3.8/site-packages/espnet_model_zoo/653d10049fdc264f694f57b49849343e/exp/asr_train_asr_transformer_e18_raw_bpe_sp/54epoch.pth"
LM_MODEL_PATH = "/home/jmpark/espnet/tools/anaconda/envs/espnet_1.7/lib/python3.8/site-packages/espnet_model_zoo/653d10049fdc264f694f57b49849343e/exp/lm_train_lm_adam_bpe/17epoch.pth"

speech2text = Speech2Text(
    **d.download_and_unpack('Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best'),
    # Decoding parameters are not included in the model file
    maxlenratio=0.0,
    minlenratio=0.0,
    beam_size=1,
    ctc_weight=0.4,
    lm_weight=0.6,
    penalty=0.0,
    nbest=1
)
# Add register hook for in encoder layers.
net = speech2text.asr_model

for name, parameter in net.named_modules():
    for i in range(18): 
        if 'encoder.encoders.'+ str(i) +'.self_attn' == name:
            parameter.register_forward_hook(attn_encoder)

# Do forward path calculation for extract image
out = speech2text(speech)

save_image(saved_encoder_self_attn_images, 'encoder_self_attn', 18, 8)
