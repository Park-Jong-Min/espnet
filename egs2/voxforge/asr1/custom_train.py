import yaml
import random
import argparse
import os

parser = argparse.ArgumentParser(description='base yaml files')

parser.add_argument('--baseyaml', required=True, help='base yaml file path')

args = parser.parse_args()

with open(args.baseyaml) as f:
    dict = yaml.load(f, Loader=yaml.FullLoader)
    n_head = dict['encoder_conf']['attention_heads']
    encl = dict['encoder_conf']['num_blocks']
    ff = dict['encoder_conf']['linear_units']
    prune_act = dict['encoder_conf']['prune_act']


for i in range(1):

    # lr = round(random.uniform(0.001, 0.005), 4)
    # ctc_weight = round(random.uniform(0.2, 0.4), 2)

    lr = 0.005
    ctc_weight = 0.3

    with open(f'{args.baseyaml[:-5]}_lr{lr}_ctc{ctc_weight}.yaml', 'w') as f:
        dict['optim_conf']['lr'] = lr
        dict['model_conf']['ctc_weight'] = ctc_weight
        yaml.dump(dict, f)

    os.system(f"./run.sh \
            {args.baseyaml[:-5]}_lr{lr}_ctc{ctc_weight}.yaml \
            TF_CUSTOM_HEAD{n_head}_ENCL{encl}_FF{ff}_{prune_act}_lr{int(lr*10000)}_ctc{int(ctc_weight*100)} \
            exp/STATS_TF_CUSTOM_HEAD{n_head}_ENCL{encl}_FF{ff}_{prune_act}_lr{int(lr*10000)}_ctc{int(ctc_weight*100)}")