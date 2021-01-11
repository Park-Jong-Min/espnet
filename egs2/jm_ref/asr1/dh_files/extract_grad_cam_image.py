import soundfile
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.asr_inference import Speech2Text

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

def save_encoder_grad_image(image_list, target_list, audio_num, n_targets, PATH):
    fig_saved_dir = PATH

    for tar in range(n_targets):
        fig = plt.figure(figsize=(18,8))
        img = image_list[tar]
        plt.imshow(img)
        plt.axis('off')

        print(f'process {target_list[tar]} target images....')

        plt.savefig(fig_saved_dir + f'head_score_audio_{audio_num}_target_{target_list[tar]}.png',
                    bbox_inches='tight',
                    dpi=100)
        plt.close()

def make_grad_cam_img_list(model, target_out, target_loss):
    model.zero_grad()

    for layer_idx in range(18):
        hook = globals()[f'hook_{layer_idx}']
        hook.target_output.grad = None

    target_out.backward(gradient=target_loss, retain_graph=True)

    out_image = torch.zeros((18, 8))

    for layer_idx in range(18):
        hook = globals()[f'hook_{layer_idx}']

        grad_out_head_view = hook.target_output.grad.view(1, -1, 8, 64)
        gap_grad = torch.mean(grad_out_head_view, dim=3).unsqueeze(3)

        feature = hook.target_output.view(1, -1, 8, 64)
        grad_cam = F.relu(torch.mul(feature, gap_grad)).mean(dim=1).mean(dim=2)

        out_image[layer_idx] = grad_cam.detach().clone()
    
    return out_image.detach().numpy()


if __name__ == "__main__":

    class Hook():
        def __init__(self, module):
            self.hook_f = module.register_forward_hook(self.hook_f_fn)
            self.target_output = None

        def hook_f_fn(self, module, input, output):
            self.target_output = input[0]

        def close(self):
            self.hook.remove()
    
    def apply_hook(model, layer_idx, module_type, attn_type):
        # module_type : 'encoder', 'decoder'
        # attn_type : 'self_attn', 'src_attn'
        for name, module in model.named_modules():
            if f'{module_type}.{module_type}s.{layer_idx}.{attn_type}.linear_out' == name:
                hook = Hook(module=module)
        
        return hook

    exp_dir = '/home/jmpark/home_data_jmpark/espnet/egs2/jm_ref/asr1/exp'

    saved_encoder_grad_cam_images = []

    TEST_DATA_PATH = "./../data/dev_clean"
    WAV_LIST_PATH = TEST_DATA_PATH + "/wav.scp"
    ANSWER_LIST_PATH = TEST_DATA_PATH + "/text"

    file_name_list = []
    speech_ans_list = []

    with open(WAV_LIST_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            num, name = line.split(' ')
            file_name_list.append(name[:-1])

    with open(ANSWER_LIST_PATH, "r") as f:
        lines = f.readlines()
        for line in lines:
            speech_ans_list.append(line[17:])

    # Prepare model
    d = ModelDownloader()

    speech2text = Speech2Text(
        **d.download_and_unpack('Shinji Watanabe/librispeech_asr_train_asr_transformer_e18_raw_bpe_sp_valid.acc.best'),
        # Decoding parameters are not included in the model file
        maxlenratio=0.0,
        minlenratio=0.0,
        beam_size=1,
        ctc_weight=1.0,
        lm_weight=0.0,
        penalty=0.0,
        nbest=1,
        out_mode="ctc"
    )

    word_num_list = []
    img_list = []

    # Add register hook for in encoder layers.
    net = speech2text.asr_model
    
    audio_num = 200 # selelct one of the wav in file_name_list
    speech, rate = soundfile.read(file_name_list[audio_num])

    for i in range(18):
        globals()[f'hook_{i}'] = apply_hook(net, i, 'encoder', 'self_attn')
    
    out, ctc_out = speech2text(speech)
    ctc_argmax = ctc_out.argmax(2)
    n_targets = 0
    mode = 'sentence'
    space = False

    createFolder(exp_dir + f'/feature_images/encoder_grad_cam/{mode}/audio_{audio_num}')

    if mode == "sentence":
        one_hot = torch.zeros_like(ctc_out)
        one_hot.scatter_(2, ctc_argmax.unsqueeze(2), 1.0)
        img = make_grad_cam_img_list(model=net, target_out=ctc_out, target_loss=one_hot)
        img_list.append(img)
        word_num_list.append(audio_num)
        
        save_encoder_grad_image(image_list=img_list, target_list=word_num_list, audio_num=audio_num,
                                n_targets=n_targets + 1, PATH=exp_dir + f'/feature_images/encoder_grad_cam/{mode}/audio_{audio_num}/')


    elif mode == "word":
        for tar in range(ctc_out.size(1)):
            if ctc_argmax[0, tar] == 0:
                if space == False:
                    continue
                else:
                    n_targets += 1
                    one_hot = torch.zeros_like(ctc_out)
                    one_hot[0, tar, ctc_argmax[0,tar].item()] = 1
                    img = make_grad_cam_img_list(model=net, target_out=ctc_out, target_loss=one_hot)
                    img_list.append(img)
                    word_num_list.append(f'{tar}_{ctc_argmax[0,tar]}')
            
            else:
                n_targets += 1
                one_hot = torch.zeros_like(ctc_out)
                one_hot[0, tar, ctc_argmax[0,tar].item()] = 1
                img = make_grad_cam_img_list(model=net, target_out=ctc_out, target_loss=one_hot)
                img_list.append(img)
                word_num_list.append(f'{tar}_{ctc_argmax[0,tar]}')
                
        save_encoder_grad_image(image_list=img_list, target_list=word_num_list, audio_num=audio_num,
                                n_targets=n_targets, PATH=exp_dir + f'/feature_images/encoder_grad_cam/{mode}/audio_{audio_num}/')
