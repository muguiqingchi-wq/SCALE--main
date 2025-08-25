import torch
import numpy as np
import random
import config
import os
from train import train_for_epoch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, CLIPModel, CLIPTokenizer

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__=='__main__':
    opt=config.parse_opt()
    torch.cuda.set_device(opt.CUDA_DEVICE)
    set_seed(opt.SEED)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    constructor='build_model'

    from dataset_process import Multimodal_Data
    import model
    train_set=Multimodal_Data(opt,tokenizer,opt.DATASET,'train',opt.SEED-1111)
    test_set=Multimodal_Data(opt,tokenizer,opt.DATASET,'test')
    
    label_list=[train_set.label_mapping_id[i] for i in train_set.label_mapping_word.keys()]
    model=getattr(model,constructor)(opt, label_list).cuda()

    train_loader=DataLoader(train_set, opt.BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader=DataLoader(test_set, opt.BATCH_SIZE, shuffle=False, num_workers=1)
    
    train_for_epoch(opt,model,train_loader,test_loader)
    
    exit(0)
