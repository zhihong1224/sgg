import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.utils import cat
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.data import get_dataset_statistics
from .utils_motifs import obj_edge_vectors, to_onehot, nms_overlaps, encode_box_info
from .utils_relation import layer_init
from .model_motifs import FrequencyBias
from .model_transformer import MultiHeadAttention



class Obj_MLPEnc(nn.Module):
    def __init__(self, config, obj_classes,in_channels):
        super(Obj_MLPEnc,self).__init__()
        self.cfg = config
        self.in_channels = in_channels
        self.obj_dim = in_channels
        self.obj_classes=obj_classes
        self.num_obj_cls=self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.nms_thresh = self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES

        self.pooler_resolution=self.cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.backbone_chs=self.cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.obj_layer = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.OBJ_LAYER
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        # the following word embedding layer should be initalize by glove.6B before using
        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(embed_vecs, non_blocking=True)

        input_size=self.backbone_chs*self.pooler_resolution**2
        self.activate_fn=nn.Tanh()
        self.sem_mlp = nn.Sequential(
            make_fc(input_size, 2048),
            self.activate_fn,
            make_fc(2048, 2048),
            self.activate_fn,     
        )
        self.lin_sem = nn.Linear(2048 + self.embed_dim, self.hidden_dim)

        if self.cfg.MODEL.ROI_BOX_HEAD.WITH_EA:
            self.app_mlp = nn.Sequential(
                make_fc(input_size, 2048),
                self.activate_fn,
                make_fc(2048, 2048),
                self.activate_fn,
            )
            self.lin_app = nn.Linear(2048, self.hidden_dim)

    def forward(self, roi_features, proposals, logger=None):
        # labels will be used during training
        use_gt_label = self.training or self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL
        obj_labels = cat([proposal.get_field("labels") for proposal in proposals], dim=0) if use_gt_label else None

        # label/logits embedding will be used as input
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL:
            obj_embed = self.obj_embed(obj_labels)
        else:
            obj_logits = cat([proposal.get_field("predict_logits") for proposal in proposals], dim=0).detach()
            obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight

        # encode objects with transformer
        roi_features=roi_features.view(roi_features.size(0),-1)
        sem_features=self.sem_mlp(roi_features)
        obj_pre_rep = cat((sem_features, obj_embed), -1)

        sem_feats = self.lin_sem(obj_pre_rep)
        sem_feats=F.tanh(sem_feats)

        if self.cfg.MODEL.ROI_BOX_HEAD.WITH_EA:
            app_features = self.app_mlp(roi_features)
            app_feats=self.lin_app(app_features)
            app_feats=F.tanh(app_feats)
        else:
            app_feats=sem_feats

        return sem_feats,app_feats,obj_labels,None  
    
    
class Obj_MLPDec(nn.Module):
    def __init__(self,config,obj_classes):
        super(Obj_MLPDec, self).__init__()
        self.cfg=config
        self.obj_classes=obj_classes
        self.num_obj_cls=self.cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES
        self.embed_dim=self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.nms_thresh=self.cfg.TEST.RELATION.LATER_NMS_PREDICTION_THRES
        # setting parameters
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'

        embed_vecs = obj_edge_vectors(self.obj_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.obj_embed = nn.Embedding(self.num_obj_cls, self.embed_dim)
        with torch.no_grad():
            self.obj_embed.weight.copy_(embed_vecs, non_blocking=True)

        hidden_dim=self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        num_obj_cls=len(obj_classes)

        self.out_obj=nn.Linear(hidden_dim,num_obj_cls)

    def forward(self,proposals,sem_feats,obj_labels):
        # predict obj_dists and obj_preds
        num_objs=[len(p) for p in proposals]
        if self.mode == 'predcls':
            obj_preds = obj_labels
            obj_dists = to_onehot(obj_preds, self.num_obj_cls)
        else:
            obj_dists = self.out_obj(sem_feats)
            use_decoder_nms = self.mode == 'sgdet' and not self.training
            if use_decoder_nms:
                boxes_per_cls = [proposal.get_field('boxes_per_cls') for proposal in proposals]
                obj_preds = self.nms_per_cls(obj_dists, boxes_per_cls, num_objs)
            else:
                obj_preds = obj_dists[:, 1:].max(1)[1] + 1
        obj_embed_out = self.obj_embed(obj_preds)
        return obj_dists,obj_preds,obj_embed_out

    def nms_per_cls(self, obj_dists, boxes_per_cls, num_objs):
        obj_dists = obj_dists.split(num_objs, dim=0)
        obj_preds = []
        for i in range(len(num_objs)):
            is_overlap = nms_overlaps(boxes_per_cls[i]).cpu().numpy() >= self.nms_thresh  # (#box, #box, #class)

            out_dists_sampled = F.softmax(obj_dists[i], -1).cpu().numpy()
            out_dists_sampled[:, 0] = -1

            out_label = obj_dists[i].new(num_objs[i]).fill_(0)

            for i in range(num_objs[i]):
                box_ind, cls_ind = np.unravel_index(out_dists_sampled.argmax(), out_dists_sampled.shape)
                out_label[int(box_ind)] = int(cls_ind)
                out_dists_sampled[is_overlap[box_ind, :, cls_ind], cls_ind] = 0.0
                out_dists_sampled[box_ind] = -1.0  # This way we won't re-sample

            obj_preds.append(out_label.long())
        obj_preds = torch.cat(obj_preds, dim=0)
        return obj_preds




class Edge_MLPEnc(nn.Module):
    def __init__(self,config,rel_classes):
        super(Edge_MLPEnc, self).__init__()
        self.cfg = config
        if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
            self.mode = 'predcls' if self.cfg.MODEL.ROI_RELATION_HEAD.USE_GT_OBJECT_LABEL else 'sgcls'
        else:
            self.mode = 'sgdet'
        self.rel_classes = rel_classes
        self.num_rel_cls = len(rel_classes)
        self.embed_dim = self.cfg.MODEL.ROI_RELATION_HEAD.EMBED_DIM
        self.hidden_dim = self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.pooling_dim=self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM

        self.dropout_rate = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.DROPOUT_RATE
        self.num_head = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.NUM_HEAD
        self.inner_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.INNER_DIM
        self.k_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.KEY_DIM
        self.v_dim = self.cfg.MODEL.ROI_RELATION_HEAD.TRANSFORMER.VAL_DIM

        embed_vecs = obj_edge_vectors(self.rel_classes, wv_dir=self.cfg.GLOVE_DIR, wv_dim=self.embed_dim)
        self.rel_embed = nn.Embedding(self.num_rel_cls, self.embed_dim)
        with torch.no_grad():
            self.rel_embed.weight.copy_(embed_vecs, non_blocking=True)

        self.resolution = self.cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        self.backbone_chs = self.cfg.MODEL.RESNETS.BACKBONE_OUT_CHANNELS
        self.activate_fn=nn.Tanh()

        input_size=self.backbone_chs*self.resolution**2
        self.union_mlp=nn.Sequential(
            make_fc(input_size, 2048),
            self.activate_fn,
            make_fc(2048, 2048),
            self.activate_fn,
        )
        self.lin_edge = nn.Linear(2048, self.hidden_dim)

        if config.MODEL.ROI_RELATION_HEAD.ONE_ATTN:   # todo: add to ablation for one layer attn----2021/12/7-----
            self.rel_sub_attn=MultiHeadAttention(n_head=self.num_head,d_model=self.hidden_dim,
                                                   d_model_kv=self.hidden_dim+self.embed_dim,d_k=self.k_dim,d_v=self.v_dim,
                                                   dropout=self.dropout_rate)
            self.rel_obj_attn = MultiHeadAttention(n_head=self.num_head, d_model=self.hidden_dim,
                                                   d_model_kv=self.hidden_dim+self.embed_dim, d_k=self.k_dim, d_v=self.v_dim,
                                                   dropout=self.dropout_rate)
        else:
            self.union_rel_attn = MultiHeadAttention(n_head=self.num_head, d_model=self.hidden_dim,
                                                     d_model_kv=self.embed_dim, d_k=self.k_dim, d_v=self.v_dim,
                                                     dropout=self.dropout_rate)
            self.rel_sub_attn = MultiHeadAttention(n_head=self.num_head, d_model=self.hidden_dim,
                                                   d_model_kv=self.hidden_dim + self.embed_dim, d_k=self.k_dim,
                                                   d_v=self.v_dim,
                                                   dropout=self.dropout_rate)
            self.rel_obj_attn = MultiHeadAttention(n_head=self.num_head, d_model=self.hidden_dim,
                                                   d_model_kv=self.hidden_dim + self.embed_dim, d_k=self.k_dim,
                                                   d_v=self.v_dim,
                                                   dropout=self.dropout_rate)

        # todo:only use vision information----2021/11/20-------------------------------------------------------------------
        # self.rel_sub_attn = MultiHeadAttention(n_head=self.num_head, d_model=self.hidden_dim,
        #                                        d_model_kv=self.hidden_dim, d_k=self.k_dim,
        #                                        d_v=self.v_dim,
        #                                        dropout=self.dropout_rate)
        # self.rel_obj_attn = MultiHeadAttention(n_head=self.num_head, d_model=self.hidden_dim,
        #                                        d_model_kv=self.hidden_dim, d_k=self.k_dim,
        #                                        d_v=self.v_dim,
        #                                        dropout=self.dropout_rate)
        self.post_cat=nn.Linear(self.hidden_dim*2,self.pooling_dim)
        layer_init(self.post_cat,xavier=True)

    def forward(self,proposals,rel_pair_idxs,union_features,obj_feats,obj_preds,obj_embed_out=None,sample_feats=None,obj_labels=None):
        # union_feats:(n_rels,f)
        # sample_feats:(n_objs,k,f)//(n_objs,num,k,f)
        num_objs=[len(p) for p in proposals]
        num_rels=[r.size(0) for r in rel_pair_idxs]
        union_feats=union_features.view(union_features.size(0),-1)
        union_feats=self.union_mlp(union_feats)
        union_feats=self.lin_edge(union_feats)   
        union_feats=F.tanh(union_feats)

        union_feats=union_feats.unsqueeze(dim=1)   # (n,1,512)
        if self.cfg.MODEL.ROI_RELATION_HEAD.ONE_ATTN:    # todo: add to ablation for one layer attn----2021/12/7-----
            union_embss=union_feats
            union_embs=union_feats.split(num_rels,dim=0)
        else:
            rel_emb=self.rel_embed.weight.expand(union_feats.size(0),self.num_rel_cls,self.embed_dim)
            union_embss,_=self.union_rel_attn(union_feats,rel_emb,rel_emb)
            union_embss=F.tanh(union_embss)
            union_embs=union_embss.split(num_rels,dim=0)
        sobj_feats=obj_feats.unsqueeze(dim=1)
        sobj_feats=sobj_feats.split(num_objs,dim=0)
        sobj_preds=obj_preds.split(num_objs,dim=0)
        # TODO: Add sample_feats------
        if sample_feats is not None:
            if len(sample_feats)==4:
                _,nums,k,f=sample_feats.shape     # todo: add 2021/11/18---
            sample_feats=sample_feats.split(num_objs,dim=0)
            obj_embed_out=obj_embed_out.split(num_objs,dim=0)
            obj_labels=obj_labels.split(num_objs,dim=0)

        pair_preds=[]
        pair_labels=[]
        prod_reps=[]
        z_prod_reps=[]
        # print('num_rels:',num_rels,'num_objs:',num_objs)
        for idx,(pair_idx,sobj_feat,union_emb,obj_pred) in enumerate(zip(rel_pair_idxs,sobj_feats,union_embs,sobj_preds)):
            if union_emb.size(0)==0:     # remove image which don't have relations,such as only contain one object
                continue
            sub_feat=sobj_feat[pair_idx[:,0]]
            obj_feat=sobj_feat[pair_idx[:,1]]
            # TODO: Add to do ablation exp------2021/11/18---
            if self.cfg.MODEL.ROI_RELATION_HEAD.WITH_ATTN:
                # TODO: ablation for attention mechanism
                # union_emb = union_emb.permute(1,0,2)
                # sub_feat = sub_feat.permute(1,0,2)
                # obj_feat = obj_feat.permute(1,0,2)
                
                sub_emb,_=self.rel_sub_attn(union_emb,sub_feat,sub_feat)
                sub_emb=F.tanh(sub_emb)
                obj_emb,_=self.rel_obj_attn(union_emb,obj_feat,obj_feat)
                obj_emb=F.tanh(obj_emb)
                #TODO: ablation for attention mechanism
                # sub_emb = sub_emb.permute(1,0,2)
                # obj_emb = obj_emb.permute(1,0,2)
                # union_emb =union_emb.permute(1,0,2)
            else:
                sub_emb=sub_feat[...,:self.hidden_dim]
                obj_emb=obj_feat[...,:self.hidden_dim]    # due to concatenate obj_embed

            # prod_reps.append(torch.cat((sub_emb.squeeze(dim=1),obj_emb.squeeze(dim=1)),dim=-1))
            prod_reps.append(torch.cat((sub_emb.squeeze(dim=1),obj_emb.squeeze(dim=1)),dim=-1))   # todo: add pos_emb--2021/11/17
            pair_preds.append(torch.stack((obj_pred[pair_idx[:,0]],obj_pred[pair_idx[:,1]]),dim=1))

            # TODO: ADD attention to sample_feats---------------
            if sample_feats is not None:
                pair_labels.append(torch.stack((obj_labels[idx][pair_idx[:, 0]], obj_labels[idx][pair_idx[:, 1]]), dim=1))
                z_sub_feat=sample_feats[idx][pair_idx[:,0]]    # (n,K,512)   aready -1~1,don't need to use tanh()
                z_obj_feat=sample_feats[idx][pair_idx[:,1]]    # (n,K,512)
                # print(obj_embed_out[idx][pair_idx[:,0]].unsqueeze(dim=1).shape,z_sub_feat.shape,)
                nb,nk=z_sub_feat.shape[0],z_sub_feat.shape[1]
                z_sub_feat=cat((z_sub_feat,obj_embed_out[idx][pair_idx[:,0]].unsqueeze(dim=1).expand(nb,nk,self.embed_dim)),dim=-1) # (n,k,712)
                z_obj_feat=cat((z_obj_feat,obj_embed_out[idx][pair_idx[:,1]].unsqueeze(dim=1).expand(nb,nk,self.embed_dim)),dim=-1)
                # TODO: Add to do ablation exp----2021/11/18---
                if self.cfg.MODEL.ROI_RELATION_HEAD.WITH_ATTN:
                    z_sub_emb,_=self.rel_sub_attn(union_emb,z_sub_feat,z_sub_feat)
                    z_obj_emb,_=self.rel_obj_attn(union_emb,z_obj_feat,z_obj_feat)
                    z_sub_emb=F.tanh(z_sub_emb)
                    z_obj_emb=F.tanh(z_obj_emb)
                else:
                    z_sub_emb=z_sub_feat.mean(dim=1,keepdim=True)[...,:self.hidden_dim]
                    z_obj_emb=z_obj_feat.mean(dim=1,keepdim=True)[...,:self.hidden_dim]

                # over_entites_1=torch.cat((sub_emb,z_obj_emb),dim=-1)    # (num_rel,1,1024)
                # over_entites_2=torch.cat((z_sub_emb,obj_emb),dim=-1)
                fake_entites=torch.cat((z_sub_emb,z_obj_emb),dim=-1)
                z_prod_reps.append(fake_entites)   # (num_rel,1,1024)

            # if sample_feats is not None:     # todo: num samples for one instance---2021/11/18---
            #     pair_labels.append(torch.stack((obj_labels[idx][pair_idx[:, 0]], obj_labels[idx][pair_idx[:, 1]]), dim=1))
            #     z_sub_feat=sample_feats[idx][pair_idx[:,0]].view(-1,k,f)    # (n*nums,K,512)   aready -1~1,don't need to use tanh()
            #     z_obj_feat=sample_feats[idx][pair_idx[:,1]].view(-1,k,f)    # (n*nums,K,512)
            #     nb=z_sub_feat.shape[0]
            #     # todo:only use visual information----don't add obj_emb----2021/11/20----------------------------------------------------
            #     z_sub_feat=cat((z_sub_feat,obj_embed_out[idx][pair_idx[:,0]].repeat(nums,1).unsqueeze(dim=1).expand(nb,k,self.embed_dim)),dim=-1) # (n*nums,k,712)
            #     z_obj_feat=cat((z_obj_feat,obj_embed_out[idx][pair_idx[:,1]].repeat(nums,1).unsqueeze(dim=1).expand(nb,k,self.embed_dim)),dim=-1)
            #     # TODO: Add to do ablation exp----2021/11/18---
            #     if self.cfg.MODEL.ROI_RELATION_HEAD.WITH_ATTN:
            #         # print(union_emb.shape,z_sub_feat.shape)
            #         z_sub_emb,_=self.rel_sub_attn(union_emb.repeat(nums,1,1),z_sub_feat,z_sub_feat)
            #         z_obj_emb,_=self.rel_obj_attn(union_emb.repeat(nums,1,1),z_obj_feat,z_obj_feat)
            #         z_sub_emb=F.tanh(z_sub_emb)
            #         z_obj_emb=F.tanh(z_obj_emb)
            #     else:
            #         z_sub_emb=z_sub_feat.mean(dim=1,keepdim=True)[...,:self.hidden_dim]
            #         z_obj_emb=z_obj_feat.mean(dim=1,keepdim=True)[...,self.hidden_dim]    # due to concatenate obj_embed, for simplity we don't use obj_embed in this ablation
            #
            #     fake_entites=torch.cat((z_sub_emb,z_obj_emb),dim=-1)    # (num_rels*nums,1,1024)
            #     z_prod_reps.append(fake_entites.view(-1,nums,fake_entites.shape[-1]))   # (num_rel,nums,1024)

        prod_reps=cat(prod_reps,dim=0)
        pair_preds=cat(pair_preds,dim=0)

        rel_feats=self.post_cat(prod_reps)
        rel_feats=F.tanh(rel_feats)

        # TODO: Add sample feats----------
        z_rel_feats=None
        if len(z_prod_reps) != 0:
            pair_labels = cat(pair_labels, dim=0)   # (all_num_rels,2)
            z_prod_reps = cat(z_prod_reps, dim=0)    # (all_num_rels,nums,1024)
            z_rel_feats=self.post_cat(z_prod_reps)   # (all_num_rels,nums,4096)
            z_rel_feats=F.tanh(z_rel_feats)

        return rel_feats,union_embss.squeeze(dim=1),pair_preds,z_rel_feats,pair_labels



class Edge_MLPDec(nn.Module):
    def __init__(self,config):
        super(Edge_MLPDec, self).__init__()
        self.cfg=config
        self.pooling_dim=self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_POOLING_DIM
        self.hidden_dim=self.cfg.MODEL.ROI_RELATION_HEAD.CONTEXT_HIDDEN_DIM
        self.use_bias=self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS
        self.num_rel_cls=self.cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES

        self.rel_compress=nn.Linear(self.pooling_dim,self.num_rel_cls)
        self.ctx_compress=nn.Linear(self.hidden_dim,self.num_rel_cls)
        layer_init(self.rel_compress,xavier=True)
        layer_init(self.ctx_compress,xavier=True)

        statistics=get_dataset_statistics(config)
        if self.use_bias:
            self.freq_bias=FrequencyBias(config,statistics)

    def forward(self,rel_feats,union_embss,pair_preds):
        rel_dists=self.rel_compress(rel_feats)    # (n_rels,51) or (n_rels*3,51)
        if self.cfg.MODEL.ROI_RELATION_HEAD.ONE_ATTN:    # todo: add one attn for ablation----2021/12/7------------
            ctx_dists=None
        else:
            ctx_dists=self.ctx_compress(union_embss)  # (n_rels,51)
        freq_dists=self.freq_bias.index_with_labels(pair_preds)  # (n_rels,51)

        if ctx_dists is not None:
            if ctx_dists.shape!=rel_dists.shape:
                nk=rel_dists.shape[0]//ctx_dists.shape[0]
                rel_dists=rel_dists.view(-1,nk,self.num_rel_cls)
                ctx_dists=ctx_dists.unsqueeze(1)
                freq_dists=freq_dists.unsqueeze(1)
            dists=rel_dists+ctx_dists
        else:
            if freq_dists.shape!=rel_dists.shape:
                nk = rel_dists.shape[0] // freq_dists.shape[0]
                rel_dists = rel_dists.view(-1, nk, self.num_rel_cls)
                freq_dists = freq_dists.unsqueeze(1)
            dists=rel_dists
        if self.use_bias:
            dists=dists+freq_dists
        dists=dists.view(-1,self.num_rel_cls)
        return dists


class CVAE(nn.Module):
    def __init__(self, in_dim=512, hid_dim=256,lat_dim=128,cls_dim=151,use_centroies=False):
        super(CVAE, self).__init__()

        self.enc_fc_1 = make_fc(in_dim + cls_dim, hid_dim)
        self.enc_fc_2 = make_fc(hid_dim, hid_dim)
        self.enc_fc_3_mu = make_fc(hid_dim, lat_dim)
        self.enc_fc_3_logvar = make_fc(hid_dim, lat_dim)

        self.dec_fc_1 = make_fc(lat_dim + cls_dim, hid_dim)
        self.dec_fc_2 = make_fc(hid_dim, hid_dim)
        self.dec_fc_3 = make_fc(hid_dim, in_dim)

        self.activation_func = F.tanh 

        self.use_centrioes=use_centroies
        if self.use_centrioes:
            self.centrioes=nn.Embedding(cls_dim,lat_dim)

    def encode(self, input, cond):
        x = torch.cat([input, cond], 1)
        out = self.enc_fc_1(x)
        out = self.activation_func(out)
        out = self.enc_fc_2(out)
        out = self.activation_func(out)
        out_mu = self.enc_fc_3_mu(out)
        out_logvar = self.enc_fc_3_logvar(out)
        return out_mu ,out_logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 *logvar)
        eps = torch.randn_like(std)
        return mu + eps *std

    def decode(self, input, cond):
        x = torch.cat([input, cond], 1)
        out = self.dec_fc_1(x)
        out = self.activation_func(out)
        out = self.dec_fc_2(out)
        out = self.activation_func(out)
        out = self.dec_fc_3(out)
        out = F.tanh(out)   
        return out

    def forward(self, input, cond):
        mu, logvar = self.encode(input, cond)
        z = self.reparameterize(mu, logvar)
        recons_x=self.decode(z,cond)
        return recons_x, mu, logvar

