import os
import torch.nn as nn
import json
import pickle as pkl
import numpy as np
import torch
import utils
from tqdm import tqdm
import config
import random
from transformers import RobertaTokenizer, RobertaModel
import sys


def load_pkl(path):
    data=pkl.load(open(path,'rb'))
    return data
    
def read_hdf5(path):
    data=h5py.File(path,'rb')
    return data

def read_csv(path):
    data=pd.read_csv(path)
    return data

def read_csv_sep(path):
    data=pd.read_csv(path,sep='\t')
    return data
    
def dump_pkl(path,info):
    pkl.dump(info,open(path,'wb'))  
    
def read_json(path):
    utils.assert_exits(path)
    data=json.load(open(path,'rb'))
    '''in anet-qa returns a list'''
    return data

def pd_pkl(path):
    data=pd.read_pickle(path)
    return data

def read_jsonl(path):
    total_info=[]
    with open(path,'rb')as f:
        d=f.readlines()
    for i,info in enumerate(d):
        data=json.loads(info)
        total_info.append(data)
    return total_info

class Multimodal_Data():
    #mem, off, harm
    def __init__(self,opt,tokenizer,dataset,mode='train',few_shot_index=0):
        super(Multimodal_Data,self).__init__()
        self.opt=opt
        self.tokenizer = tokenizer
        self.mode=mode
        
        self.num_ans=self.opt.NUM_LABELS
        self.num_sample = self.opt.NUM_SAMPLE
        #maximum length for a single sentence
        self.length=self.opt.LENGTH
        self.total_length=self.opt.TOTAL_LENGTH

        if len(self.opt.ASK_CAP.split(','))>=1:
            self.ask_cap=True
        else:
            self.ask_cap=False
        

        self.add_ent=self.opt.ADD_ENT
        self.add_dem=self.opt.ADD_DEM
        self.num_meme_cap=self.opt.NUM_MEME_CAP
        print ('Adding exntity information?',self.add_ent)
        print ('Adding demographic information?',self.add_dem)
        self.fine_grind=self.opt.FINE_GRIND
        print ('Using target information?',self.fine_grind)
        
        self.label_mapping_word={0:self.opt.POS_WORD,
                                     1:self.opt.NEG_WORD}
        self.template="*<s>**sent_0*.*_It_was*label_**</s>*"
            
            
        self.label_mapping_id={}
        for label in self.label_mapping_word.keys():
            mapping_word=self.label_mapping_word[label]
            assert len(tokenizer.tokenize(' ' + self.label_mapping_word[label])) == 1
            self.label_mapping_id[label] = \
            tokenizer._convert_token_to_id(
                tokenizer.tokenize(' ' + self.label_mapping_word[label])[0])
            print ('Mapping for label %d, word %s, index %d' % 
                   (label,mapping_word,self.label_mapping_id[label]))

        self.template_list=self.template.split('*')
        print('Template:', self.template)
        print('Template list:',self.template_list)
        self.special_token_mapping = {
            '<s>': tokenizer.convert_tokens_to_ids('<s>'),
            '<mask>': tokenizer.mask_token_id, 
            '<pad>': tokenizer.pad_token_id, #1 for roberta
            '</s>': tokenizer.convert_tokens_to_ids('<\s>') 
        }
        

        self.support_examples=self.load_entries('train')
        print ('Length of supporting example:',len(self.support_examples))

        self.entries=self.load_entries(mode)

        print ('The length of the dataset for:',mode,'is:',len(self.entries))

    def load_entries(self,mode):
        if self.opt.FEW_SHOT and mode=='train':
            path=os.path.join(self.opt.DATA,
                              'domain_splits',
                              self.opt.DATASET+'_'+str(self.num_shots)+'_'+self.few_shot_index+'.json')
        else:     
            path=os.path.join(self.opt.DATA,
                              'new_data',
                              self.opt.DATASET+'_P_'+mode+'.json')
            img_fearure_path = os.path.join(self.opt.DATA,
                              'new_data',
                              self.opt.DATASET + '_img_' + mode+'.pkl')


        data=read_json(path)
        if self.opt.CAP_TYPE == 'caption':
            cap_path=os.path.join(self.opt.CAPTION_PATH,
                                self.opt.DATASET+'_'+self.opt.PRETRAIN_DATA,
                                self.opt.IMG_VERSION+'_captions.pkl')
        elif self.opt.CAP_TYPE == 'vqa':
            cap_path=os.path.join(self.opt.CAPTION_PATH, '../Vqa-Captions/Captions',
                                  self.opt.DATASET,
                                  mode+'_generic.pkl')
            if self.opt.ASK_CAP!='':
                questions=self.opt.ASK_CAP.split(',')
                result_files={q:load_pkl(os.path.join(self.opt.CAPTION_PATH,
                    '../Vqa-Captions/' + self.opt.LONG + 'Captions',
                    self.opt.DATASET,
                    mode+'_' + q + '.pkl')) 
                              for q in questions}
                print (len(result_files))
                valid=['valid_person','valid_animal']
                for v in valid:
                    result_files[v]=load_pkl(os.path.join(self.opt.CAPTION_PATH,
                        '../Vqa-Captions/' + self.opt.LONG + 'Captions',
                        self.opt.DATASET,
                        mode+'_' + v + '.pkl'))
        
        captions=load_pkl(cap_path)
        img_features = load_pkl(img_fearure_path)
        entries=[]
        text_list = []
           
        for k,row in enumerate(data):
            label=row['label']
            img=row['img']
            
            if self.opt.CAP_TYPE=='caption':
                cap=captions[img.split('.')[0]][:-1]
            elif self.opt.CAP_TYPE=='vqa' and self.ask_cap:
                cap=captions[img]
                ext=[]
                person_flag=True
                animal_flag=True
                person=result_files['valid_person'][row['img']].lower()
                if person.startswith('no'):
                    person_flag=False
                animal=result_files['valid_animal'][row['img']].lower()
                if animal.startswith('no'):
                    animal_flag=False
                    
                for q in questions:
                    if person_flag==False and q in ['race','gender',
                                                    'country','valid_disable']:
                        continue
                    if animal_flag==False and q=='animal':
                        continue
                    if q in ['valid_person','valid_animal']:
                        continue
                        
                    info=result_files[q][row['img']]
                    if q=='valid_disable':
                        if info.startswith('no'):
                            continue
                        else:
                            ext.append('there is a disabled person')
                    else:
                        ext.append(info)

                if self.num_meme_cap>0:
                    pnp_cap_path=os.path.join(self.opt.CAPTION_PATH,
                                              '../Vqa-Captions/pnp-captions',
                                              self.opt.DATASET,img+'.json')
                    if os.path.exists(pnp_cap_path):
                        caps=read_json(pnp_cap_path)
                        ext.extend(caps[:self.num_meme_cap])
                    else:
                        ext.extend([cap]*self.num_meme_cap)

                ext=' . '.join(ext)
                cap=cap+' . '+ext
            
            
            
            sent=row['clean_sent']
            img_feat = img_features[img.split('.')[0]] 
            cap = cap +' . '+sent+' . '
            ''' method_1 '''
            if self.add_ent:
                sent = row['entity'] + ' . '+ sent + ' . '
            if self.add_dem:
                sent = row['race'] + ' . ' + sent + ' . '
            
            sent = sent + ' . ' + cap + ' . '

            if self.add_ent:
                cap = row['entity'] + ' . ' + cap + ' . '
            if self.add_dem:
                cap = row['race'] + ' . ' + cap + ' . '

            entry={              
                "img_feat": img_feat,
                'sent': sent.strip(),
                'cap': cap.strip(),
                'label': label,
                'img': img,
            }
            entries.append(entry)   
        

        
        return entries
    
    def enc(self,text):
        return self.tokenizer.encode(text, add_special_tokens=True)
                
    def __getitem__(self,index):
        entry=self.entries[index]
        exps=[]
        exps.append(entry)
        sent_input_ids = []
        caption_input_ids = []

        sent_attention_mask = []   
        caption_attention_mask = []
        
        length = self.length
        sent_input_ids += self.enc(entry["sent"])
        caption_input_ids += self.enc(entry["cap"])

        sent_input_ids = sent_input_ids[:length]
        caption_input_ids = caption_input_ids[:length]

        sent_attention_mask += [1 for i in range(len(sent_input_ids))]
        caption_attention_mask += [1 for i in range(len(caption_input_ids))]
        

        while len(sent_input_ids) < self.total_length:
            sent_input_ids.append(self.special_token_mapping['<pad>'])
            sent_attention_mask.append(0)

        if len(sent_input_ids) > self.total_length:
            sent_input_ids = sent_input_ids[:self.total_length]
            sent_attention_mask = sent_attention_mask[:self.total_length]
            

        while len(caption_input_ids) < self.total_length:
            caption_input_ids.append(self.special_token_mapping['<pad>'])
            caption_attention_mask.append(0)

        if len(caption_input_ids) > self.total_length:
            caption_input_ids = caption_input_ids[:self.total_length]
            caption_attention_mask = caption_attention_mask[:self.total_length]
            
        label=torch.tensor(entry['label'])

        target=torch.from_numpy(np.zeros((self.num_ans),dtype=np.float32))
        target[label]=1.0
        
        cap_tokens = torch.Tensor(caption_input_ids)
        sent_tokens = torch.Tensor(sent_input_ids)
        cap_mask = torch.Tensor(caption_attention_mask)
        sent_mask = torch.Tensor(sent_attention_mask)
        batch={
            'cap_mask':cap_mask,
            'sent_mask': sent_mask,
            'target':target,
            'cap_tokens':cap_tokens,
            'sent_tokens': sent_tokens,
            'label':label,
            'img_feat':entry["img_feat"],
            'img': entry["img"]
        }
        if self.fine_grind:
            batch['attack']=torch.Tensor(entry['attack'])
        return batch
        
    def __len__(self):
        return len(self.entries)
    
