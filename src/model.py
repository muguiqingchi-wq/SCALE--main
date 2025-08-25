import torch
import torch.nn as nn
from transformers import RobertaForMaskedLM
from transformers import RobertaTokenizer, RobertaModel
from AlignNets import AlignSubNet
import pickle
from transformers_encoder.transformer import TransformerEncoder
from easydict import EasyDict
import config
import torch.nn.functional as F




class MAG(nn.Module):
    def __init__(self, opt, args):
        super(MAG,self).__init__()
        
        self.alignNet = AlignSubNet(args, 'ctc')        
        
        self.W_h_img = nn.Linear(args.img_feat_dim + args.text_feat_dim, args.text_feat_dim)
        self.W_h_cap = nn.Linear(args.cap_feat_dim + args.text_feat_dim, args.text_feat_dim)

        self.W_img = nn.Linear(args.img_feat_dim, args.text_feat_dim)
        self.W_cap = nn.Linear(args.cap_feat_dim, args.text_feat_dim)    
        
        self.LayerNorm = nn.LayerNorm(args.hidden_size)
    
        self.beta_shift = args.beta_shift
        self.dropout = nn.Dropout(args.dropout_prob)

    def forward(self, text_embedding, img_feat, cap_feat):
        eps = 1e-6
                
        aligned_text_embedding, aligned_img, aligned_cap = self.alignNet(text_embedding,img_feat,cap_feat)
        
        weight_img = F.relu(self.W_h_img(torch.cat((aligned_img, aligned_text_embedding), dim=-1)))
        weight_cap = F.relu(self.W_h_cap(torch.cat((aligned_cap, aligned_text_embedding), dim=-1)))
        
        h_m = weight_img * self.W_img(aligned_img) + weight_cap * self.W_cap(aligned_cap)
        
        em_norm = aligned_text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)
        
        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).to(aligned_text_embedding.device)
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)
        
        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift
        
        ones = torch.ones(thresh_hold.shape, requires_grad=True).to(aligned_text_embedding.device)
        
        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)
        
        img_cap_embedding = alpha * h_m
        
        embedding_output = self.dropout(
            self.LayerNorm(img_cap_embedding + aligned_text_embedding)
        )
        
        return embedding_output

        
    
class RobertaPromptModel(nn.Module):
    
    def __init__(self,label_list):
        super(RobertaPromptModel, self).__init__()
        opt=config.parse_opt()
        torch.cuda.set_device(opt.CUDA_DEVICE)
        self.label_word_list=label_list

        self.model_roberta = RobertaModel.from_pretrained('roberta-large')
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.gamma = nn.Parameter(torch.ones(1024) * 1e-4).to(opt.CUDA_DEVICE)
        self.loss_BCE = nn.BCEWithLogitsLoss()
        

        self.prompt_conv = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        
        self.cap_len = 256
        
        self.img_len = 49
        self.embed_dim = 1024
        self.text_dim = 1024
        self.img_dim = 768
        self.dropout =  nn.Dropout(0.2)
        
        self.ctx = self.init_ctx(self.cap_len,self.text_dim)
        
        self.output_layer = nn.Sequential(
            nn.LayerNorm(self.embed_dim * 2),            
            nn.AdaptiveAvgPool2d((1, 2048)),
            nn.Flatten(),
            nn.Linear(2048, 1024),
            nn.ReLU(),              
            nn.Linear(1024, 512),
            nn.ReLU(),            
            nn.Dropout(0.2), 
            nn.Linear(512, 1),
            
            
        )

        
        self.caption_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )

        self.image_proj = nn.Sequential(
            nn.LayerNorm(self.img_dim),
            nn.Linear(self.img_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
        )


        self.text_proj = nn.Sequential(
            nn.LayerNorm(self.text_dim),
            nn.Linear(self.text_dim, self.embed_dim),
        )
        
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.text_dim)
        )
        
        
        self.trans_a_with_l = TransformerEncoder(embed_dim=self.embed_dim,
                                num_heads=8,                                
                                layers=5,
                                attn_dropout=0.1,
                                relu_dropout=0,
                                res_dropout=0.1,
                                embed_dropout=0.2,
                                attn_mask=True)
        
        self.args = EasyDict({
            "text_feat_dim": self.text_dim,
            "img_feat_dim": self.img_dim,
            "cap_feat_dim": self.text_dim,
            "max_cons_seq_length": self.cap_len,
            "img_seq_len": self.img_len,
            "cap_seq_len": self.cap_len,
            "shared_dim": self.embed_dim,
            "eps": 1e-09,
            'dropout_prob': 0.5,
            'beta_shift': 0.006,
            'hidden_size': 1024
        })
        
        self.alignNet = AlignSubNet(self.args,'sim')        
        self.MAG = MAG(opt, self.args)                
        self.ctx = self.init_ctx(self.cap_len,self.text_dim)
        self.ctx = self.ctx.to(opt.CUDA_DEVICE)
        
        
    def forward(self,cap_tokens,sent_tokens,cap_mask,sent_mask,img_feat,labels,feat=None):
        opt=config.parse_opt()
        torch.cuda.set_device(opt.CUDA_DEVICE)
        self.cap_tokens = cap_tokens
        self.sent_tokens = sent_tokens
        self.cap_mask = cap_mask
        self.sent_mask = sent_mask
        self.img_feat = img_feat
        
        temp_sent_out = self.model_roberta(self.sent_tokens,self.sent_mask)
        
        temp_cap_out = self.model_roberta(self.cap_tokens,self.cap_mask)

        self.cap_last_hidden = temp_cap_out.last_hidden_state
        
        self.sent_last_hidden = temp_sent_out.last_hidden_state
        
        batch_ctx = self.ctx.unsqueeze(0).repeat(self.cap_last_hidden.shape[0], 1, 1)

        _, aligned_image, aligned_caption  = self.alignNet(batch_ctx, self.img_feat, self.cap_last_hidden)    
        aligned_caption = self.caption_proj(aligned_caption)        
        aligned_image = self.image_proj(aligned_image)
        
        batch_ctx = self.text_proj(batch_ctx)        
        
        generated_ctx = self.trans_a_with_l(batch_ctx.permute(1, 0, 2), aligned_caption.permute(1, 0, 2), aligned_image.permute(1, 0, 2)).permute(1, 0, 2)
        generated_ctx = batch_ctx + self.out_proj(generated_ctx) * self.gamma
        
        
        fused_sent_feature = self.MAG(self.sent_last_hidden, img_feat, self.cap_last_hidden)        
        fusion_feat = torch.cat((fused_sent_feature, generated_ctx), dim=2)
        fusion_feat_temp = fusion_feat.unsqueeze(1)
        fusion_feat_temp = self.prompt_conv(fusion_feat_temp)
        fusion_feat_temp = fusion_feat_temp.squeeze(1)
        fusion_feat = fusion_feat + fusion_feat_temp
        logits = self.output_layer(fusion_feat)

        
        loss = self.loss_BCE(logits,labels)
        
        return logits, loss


    def init_ctx(self,seq_len,text_feat_dim):
        ctx = torch.empty(seq_len,text_feat_dim, dtype=torch.float)
        nn.init.trunc_normal_(ctx)
        return ctx
        
        
    
def build_model(opt,label_list):  
    print (label_list)
    return RobertaPromptModel(label_list)



    