#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 11:17:06 2022

@author: mheinzinger
"""


import argparse
import time
from pathlib import Path
from typing import Union

import numpy as np
import torch
from torch import nn

from urllib import request
import shutil

# Huggingface Transformer models:
from transformers import T5EncoderModel, T5Tokenizer
from transformers import GPT2LMHeadModel , GPT2Tokenizer

# TMbed specifics:
from tmbed_predictor import Predictor
from tmbed_viterbi import Decoder

from tqdm import tqdm

import h5py

import onnxruntime as ort
from onnxruntime.capi.onnxruntime_pybind11_state import NoSuchFile



def get_device(device: Union[None, str, torch.device] = None) -> torch.device:
    """Returns what the user specified, or defaults to the GPU,
    with a fallback to CPU if no GPU is available."""
    if isinstance(device, torch.device):
        return device
    elif device:
        return torch.device(device)
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def read_h5_embeddings(file_path):
    with h5py.File(file_path, 'r') as embeddings_file:
        id2emb = {
            embeddings_file[idx].attrs["original_id"]: np.array(embedding)
            for idx, embedding in embeddings_file.items()
        }
    return id2emb


# At the beginning, there was no device
device = None

#### Architectures
# ProtTucker predicts structural classes according to CATH
# https://github.com/Rostlab/EAT/blob/main/eat.py#L25
class TuckerFNN(nn.Module):
    def __init__(self):
        super(TuckerFNN, self).__init__()

        self.tucker = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
        )

    def single_pass(self, x):
        return self.tucker(x)

    def forward(self, X): # only needed during training
        ancor = self.single_pass(X[:, 0, :])
        pos   = self.single_pass(X[:, 1, :])
        neg   = self.single_pass(X[:, 2, :])
        return (ancor, pos, neg)


# LightAttention for subcellular localization prediction:
# https://github.com/HannesStark/protein-localization/blob/master/models/light_attention.py
class LightAttention(nn.Module):
    def __init__(self, embeddings_dim=1024, output_dim=11, dropout=0.25, kernel_size=9, conv_dropout: float = 0.25):
        super(LightAttention, self).__init__()

        self.feature_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                             padding=kernel_size // 2)
        self.attention_convolution = nn.Conv1d(embeddings_dim, embeddings_dim, kernel_size, stride=1,
                                               padding=kernel_size // 2)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(conv_dropout)

        self.linear = nn.Sequential(
            nn.Linear(2 * embeddings_dim, 32),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.BatchNorm1d(32)
        )

        self.output = nn.Linear(32, output_dim)

    def forward(self, x: torch.Tensor, mask, **kwargs) -> torch.Tensor:
        """
        Args:
            x: [batch_size, embeddings_dim, sequence_length] embedding tensor that should be classified
            mask: [batch_size, sequence_length] mask corresponding to the zero padding used for the shorter sequences in the batch. All values corresponding to padding are False and the rest is True.
        Returns:
            classification: [batch_size,output_dim] tensor with logits
        """
        o = self.feature_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        o = self.dropout(o)  # [batch_gsize, embeddings_dim, sequence_length]
        attention = self.attention_convolution(x)  # [batch_size, embeddings_dim, sequence_length]
        # mask out the padding to which we do not want to pay any attention (we have the padding because the sequences have different lenghts).
        # This padding is added by the dataloader when using the padded_permuted_collate function in utils/general.py
        attention = attention.masked_fill(mask[:, None, :] == 0, torch.tensor(-1e+4))

        # code used for extracting embeddings for UMAP visualizations
        # extraction =  torch.sum(x * self.softmax(attention), dim=-1)
        # extraction = self.id0(extraction)

        o1 = torch.sum(o * self.softmax(attention), dim=-1)  # [batchsize, embeddings_dim]
        o2, _ = torch.max(o, dim=-1)  # [batchsize, embeddings_dim]
        o = torch.cat([o1, o2], dim=-1)  # [batchsize, 2*embeddings_dim]
        o = self.linear(o)  # [batchsize, 32]
        return self.output(o)  # [batchsize, output_dim]


class BindPredictCNN(torch.nn.Module):
    # Parameters:
    # https://github.com/Rostlab/bindPredict/blob/e9f1f33c5b614966fbf7d85b79f856b68ca495ad/develop_bindEmbed21DL.py#L34
    def __init__(self, in_channels=1024, feature_channels=128, kernel_size=5, stride=1, padding=2, dropout=0.7):
        super(BindPredictCNN, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=in_channels, out_channels=feature_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding),
            torch.nn.ELU(),
            torch.nn.Dropout(dropout),

            torch.nn.Conv1d(in_channels=feature_channels, out_channels=3, kernel_size=kernel_size,
                            stride=stride, padding=padding),
        )

    def forward(self, x):
        x = self.conv1(x)
        return torch.squeeze(x)

# CNN for secondary structure prediction in 3- and 8-states (see ProtTrans)
# 3- and 8-states refer to DSSP definitions 
class SecStructCNN( nn.Module ):
    def __init__( self ):
        super(SecStructCNN, self).__init__()
        # This is only called "elmo_feature_extractor" for historic reason
        # CNN weights are trained on ProtT5 embeddings
        self.elmo_feature_extractor = nn.Sequential(
                        nn.Conv2d( 1024, 32, kernel_size=(7,1), padding=(3,0) ), # 7x32
                        nn.ReLU(),
                        nn.Dropout( 0.25 ),
                        )
        n_final_in = 32
        self.dssp3_classifier = torch.nn.Sequential(
                        nn.Conv2d( n_final_in, 3, kernel_size=(7,1), padding=(3,0)) # 7
                        )
        
        self.dssp8_classifier = torch.nn.Sequential(
                        nn.Conv2d( n_final_in, 8, kernel_size=(7,1), padding=(3,0))
                        )
        self.diso_classifier = torch.nn.Sequential(
                        nn.Conv2d( n_final_in, 2, kernel_size=(7,1), padding=(3,0))
                        )
        

    def forward( self, x):
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0,2,1).unsqueeze(dim=-1) 
        x       = self.elmo_feature_extractor(x) # OUT: (B x 32 x L x 1)
        d3_Yhat = self.dssp3_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 3)
        d8_Yhat = self.dssp8_classifier( x ).squeeze(dim=-1).permute(0,2,1) # OUT: (B x L x 8)
        return d3_Yhat, d8_Yhat


# CNN for conservation prediction in 9 classes [0-8] (see VESPA). 
# See ConSeq for conservation class definitions. 0=highly_variable; 8=highly_conserved
class ConservationCNN( nn.Module ):
    def __init__( self, pretrained_model=None ):
        super(ConservationCNN, self).__init__()
        n_features= 1024
        bottleneck_dim = 32
        n_classes = 9
        dropout_rate = 0.25
        self.classifier = nn.Sequential(
                        nn.Conv2d( n_features, bottleneck_dim, kernel_size=(7,1), padding=(3,0) ), # 7x32
                        nn.ReLU(),
                        nn.Dropout( dropout_rate ),
                        nn.Conv2d( bottleneck_dim, n_classes, kernel_size=(7,1), padding=(3,0))
                        )

    def forward( self, x):
        '''
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of classes (9 for conservation)
        '''
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0,2,1).unsqueeze(dim=-1) 
        Yhat_consurf = self.classifier(x) # OUT: Yhat_consurf = (B x N x L x 1)
        # IN: (B x N x L x 1); OUT: ( B x L x N )
        Yhat_consurf = Yhat_consurf.squeeze(dim=-1).permute(0,2,1)
        return Yhat_consurf


# Define CNN structure for SETH continuous disorder prediction (CheZOD-scores)
class DisorderCNN( nn.Module ):
    def __init__( self ):
        super(DisorderCNN, self).__init__()
        self.n_classes = 1
        bottleneck_dim = 28       
        self.classifier = nn.Sequential(
                        #summarize information from 5 neighbouring amino acids (AAs) 
                        #padding: dimension corresponding to AA number does not change
                        nn.Conv2d( 1024, bottleneck_dim, kernel_size=(5,1), padding=(2,0) ), 
                        nn.Tanh(),
                        nn.Conv2d( bottleneck_dim, self.n_classes, kernel_size=(5,1), padding=(2,0))
                        )

    def forward( self, x):
        '''
            L = protein length
            B = batch-size
            F = number of features (1024 for embeddings)
            N = number of output nodes (1 for disorder, since predict one continuous number)
        '''
        # IN: X = (B x L x F); OUT: (B x F x L, 1)
        x = x.permute(0,2,1).unsqueeze(dim=-1) 
        Yhat = self.classifier(x) # OUT: Yhat_consurf = (B x N x L x 1)
        # IN: (B x N x L x 1); OUT: ( B x L x N )
        Yhat = Yhat.squeeze(dim=-1).permute(0,2,1)
        return Yhat

#### Residue-level Predictors
# https://github.com/Rostlab/VESPA
class ConservationPredictor():
    def __init__(self, model_dir, use_onnx=False):
        if use_onnx:
            model_path = f'{model_dir}/conservation_onnx/conservation.onnx'
            self.model = ort.InferenceSession(model_path)
        else:
            self.model = self.load_model(model_dir)

    def load_model(self, model_dir):
      checkpoint_p = model_dir / "conservation_checkpoint.pt"
      weights_link = "http://data.bioembeddings.com/public/embeddings/feature_models/prott5cons/checkpoint.pt"
      model = ConservationCNN()
      return load_model(model, weights_link, checkpoint_p)

    def write_predictions(self, predictions, out_dir):
        out_p = out_dir / "conservation_pred.txt"
        with open(out_p, 'w+') as out_f:
            out_f.write( '\n'.join( 
                [ "{}".format( 
                    ''.join( [str(j) for j in yhat] )) 
                    for yhat in predictions
                    ] 
                    ) )
        return None
    
# https://github.com/BernhoferM/TMbed
class TMbed():
    
    def __init__(self,model_dir, use_onnx_model=False):

        if use_onnx_model:
            model_dir = f'{model_dir}/tmbed_onnx'
            if not Path(model_dir).is_dir():
                print('Tmbed onnx model does not exist.')
            self.model = self.load_model_onnx(model_dir)
        else:
            model_dir = model_dir / "tmbed"
            if  not model_dir.is_dir():
                model_dir.mkdir()
            self.model = self.load_model(model_dir)

        self.decoder = Decoder().to(device)

    def load_model(self,model_dir):
        model_names = ["cv_0.pt","cv_1.pt","cv_2.pt","cv_3.pt","cv_4.pt"]
        models = []
        for model_file in model_names:
            model = Predictor()
            weights_link = "http://data.bioembeddings.com/public/embeddings/feature_models/tmbed/" + model_file
            checkpoint_p = model_dir / model_file
            # TMbed has a normalization layer --> cast to fp32 for stability
            models.append( load_model(model,weights_link,checkpoint_p,state_dict="model").float() )

        return models

    def load_model_onnx(self, model_dir):
        models = []
        for onnx_file in Path(model_dir).iterdir():
            onnx_model = ort.InferenceSession(onnx_file)
            models.append(onnx_model)
        return models


    def pred2label(self):
        return {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'S', 5: 'i', 6: 'o'}
    
    def write_predictions(self,predictions,out_dir):
        out_p = out_dir / "membrane_tmbed.txt"
        class_mapping = self.pred2label()
        
        with open(out_p, 'w+') as out_f:
            out_f.write( '\n'.join( 
                [ "{}".format( 
                    ''.join( [class_mapping[j] for j in yhat] )) 
                    for yhat in predictions
                    ] 
                    ) )
        return None
       
# https://github.com/DagmarIlz/SETH
class SETH():
    def __init__(self,model_dir, use_onnnx_model=False):
        if use_onnnx_model:
            model_path = f'{model_dir}/seth_onnx/seth.onnx'
            if not Path(model_dir).is_dir():
                print(f'SETH onnx model does not exist or is not at the correct location ({model_dir}.')
            self.model = ort.InferenceSession(model_path)
        else:
            model_dir = model_dir / "seth"
            if not model_dir.is_dir():
                model_dir.mkdir()
            self.model = self.load_model(model_dir)
        
    def load_model(self,model_dir):
        checkpoint_p = model_dir / "seth_checkpoint.pt"
        # Weight link doens't work -> weight was downloaded from the original repo
        # https://github.com/DagmarIlz/SETH/blob/main/CNN/CNN.pt
        weights_link = "https://rostlab.org/~deepppi/SETH_CNN.pt"
        model = DisorderCNN()
        return load_model(model, weights_link, checkpoint_p)
    
    def write_predictions(self,predictions,out_dir):
        out_p = out_dir / "seth_disorder_pred.csv"
        with open(out_p, 'w+') as out_f:
            out_f.write( '\n'.join( 
                        [ "{}".format( ', '.join( [str(j) for j in Zscore] )) 
                         for Zscore in predictions
                        ] 
                        ) 
                )
        return None

    
# https://github.com/Rostlab/bindPredict
class BindPredict():
    
    def __init__(self,model_dir, use_onnx_model=False):
        if use_onnx_model:
            model_dir = model_dir / "bindpredict_onnx"
            if  not Path(model_dir).is_dir():
                print(f'BindPredict onnx model does not exist or is not at the correct location ({model_dir}.')
            self.model = self.load_models_onnx(model_dir=model_dir)
        else:
            model_dir = model_dir / "bindpredict"
            if  not model_dir.is_dir():
                model_dir.mkdir()
            self.model = self.load_model(model_dir)

    def load_models_onnx(self, model_dir):
        models = []
        for onnx_file in Path(model_dir).iterdir():
            onnx_model = ort.InferenceSession(onnx_file)
            models.append(onnx_model)
        return models

    def load_model(self,model_dir):
        model_names = ["checkpoint1.pt","checkpoint2.pt","checkpoint3.pt","checkpoint4.pt","checkpoint5.pt"]
        models = []

        for model_file in model_names:
            model = BindPredictCNN()
            weights_link = "http://data.bioembeddings.com/public/embeddings/feature_models/bindembed21/" + model_file
            checkpoint_p = model_dir / model_file
            models.append( load_model(model,weights_link,checkpoint_p) )

        return models

    def binding_classes(self):
        return {0: ('metal',"M"), 1: ('nucleic',"N"), 2: ('small',"S")}
    
    def write_predictions(self,predictions,out_dir):
        for idx, (binding_type, bind_short) in self.binding_classes().items():
            out_p = out_dir / "binding_bindEmbed_{}_pred.txt".format(binding_type)
            
            with open(out_p, 'w+') as out_f:
                out_f.write( '\n'.join( 
                    [ "{}".format( 
                        ''.join( [ bind_short if j==1 else "-" for j in yhat[:,idx]] )) 
                        for yhat in predictions
                        ] 
                        ) )
        return None
    
class SecStructPredictor():
    def __init__(self, model_dir, use_onnx=False):
        if use_onnx:
            model_path = f'{model_dir}/sec_struct_onnx/secstruct.onnx'
            self.model = ort.InferenceSession(model_path)
        else:
            model_dir = model_dir / "prott5_sec_struct"
            if  not model_dir.is_dir():
                model_dir.mkdir()
            self.model = self.load_model(model_dir)
        
    def load_model(self, model_dir):
      checkpoint_p = model_dir / "secstruct_checkpoint.pt"
      weights_link = "http://data.bioembeddings.com/public/embeddings/feature_models/t5/secstruct_checkpoint.pt"
      model = SecStructCNN()
      return load_model(model, weights_link, checkpoint_p)

    def label_mapping_3states(self):
        return {0:"H",1:"E",2:"L"} 
    
    def label_mapping_8states(self):
        return { idx : state for idx, state in enumerate("GHIBESTC") }

    def write_predictions(self, predictions, out_dir, dssp3=True):
        class_mapping=self.label_mapping_3states() if dssp3 else self.label_mapping_8states()
        out_name = "dssp3_pred.txt" if dssp3 else "dssp8_pred.txt"
        out_p = out_dir / out_name
        with open(out_p, 'w+') as out_f:
            out_f.write( '\n'.join( 
                [ "{}".format( 
                    ''.join( [class_mapping[j] for j in yhat] )) 
                    for yhat in predictions
                    ] 
                    ) )
        return None

#### EAT-based methods (embedding-based annotation transfer)
class GoPredSim():
    def __init__(self,model_dir,onto):
        model_dir = model_dir / "goPredSim"
        if not model_dir.is_dir():
            model_dir.mkdir()
            
        ONTOS = { "bpo" : "goa_annotations_2022_bpo.txt",
                  "mfo" : "goa_annotations_2022_mfo.txt",
                  "cco" : "goa_annotations_2022_cco.txt"
                  }
        self.onto=onto
        file_name = ONTOS[onto]
        label_p = model_dir / file_name
        self.lookup_labels = self.read_annotation(label_p)
        self.lookup_ids, self.lookup_embs = self.read_embeddings(model_dir)

        
    def read_annotation(self, label_p):
        go_annotations=dict()

        if not label_p.is_file():
            embedding_link = "http://data.bioembeddings.com/public/embeddings/reference/goa/" + label_p.name
            download_file(embedding_link,label_p)

        with open(label_p,'r') as read_in:
            next(read_in) # skip header
            for line in read_in:
                splitted_line = line.strip().split(";")
                identifier = splitted_line[0]
                go_terms = splitted_line[1]
                if identifier not in go_annotations:
                    go_annotations[identifier]=set()
                go_annotations[identifier].add(go_terms)

        return go_annotations
    
    def write_predictions(self,predictions,out_dir):
        out_p = out_dir / "goPredSim_GO_{}_pred.csv".format(self.onto)
        with open(out_p,"w+") as out_f:
            out_f.write("\n".join( [ lookup_id + "\t" + ",".join(go_anno)  + "\t{:.3f}".format( nn_dist )
                                     for lookup_id, go_anno, nn_dist in predictions
                                     ]
                                )
                    )
        return None

    
    def read_embeddings(self,goPredSim_dir):
        emb_file_name = "prott5_reference_embeddings.npy"
        emb_local_p = goPredSim_dir / emb_file_name
        if not emb_local_p.is_file():
            weights_link = "https://rostlab.org/~deepppi/goPredSim/" + emb_file_name
            download_file(weights_link,emb_local_p)
        
        embs = torch.from_numpy( np.load(emb_local_p) )
        
        id_file_name = "prott5_reference_embeddings.txt"
        id_local_p = goPredSim_dir / id_file_name
        if not id_local_p.is_file():
            weights_link = "https://rostlab.org/~deepppi/goPredSim/" + id_file_name
            download_file(weights_link,id_local_p)
            
        with open(id_local_p,'r') as in_f:
            ids = [line.strip() for line in in_f ]
    
        # extract only those embeddings that have annotations for a given part of the GO-ontologies
        filtered_ids = list()
        filtered_idxs = list()
        for i, identifier in enumerate(ids):
            if identifier in self.lookup_labels:
                filtered_ids.append(identifier)
                filtered_idxs.append(i)
                
        # add extra empty dimension to embeddings to allow for batch-comp of pairwise distances 
        return filtered_ids, embs[filtered_idxs].unsqueeze(dim=0).to(device)
    

class ProtTucker():
    def __init__(self,model_dir):
        model_dir = model_dir / "prottucker"
        if  not model_dir.is_dir():
            model_dir.mkdir()
        self.model = self.load_model(model_dir)
        self.lookup_labels = self.read_annotation(model_dir)
        self.lookup_ids, self.lookup_embs = self.read_embeddings(model_dir)
        
    def load_model(self,model_dir):
        checkpoint_p = model_dir / "tucker_weights.pt"
        weights_link = "http://rostlab.org/~deepppi/embedding_repo/embedding_models/ProtTucker/ProtTucker_ProtT5.pt"
        model = TuckerFNN()
        return load_model(model,weights_link,checkpoint_p)
    
    def read_annotation(self,model_dir):
        label_p = model_dir / "cath_v430_dom_seqs_S100_161121_labels.txt" 

        if not label_p.is_file():
            embedding_link = "https://rostlab.org/~deepppi/eat_dbs/cath_v430_dom_seqs_S100_161121_labels.txt"
            download_file(embedding_link,label_p)

        with open(label_p, 'r') as in_f:
            # protein-ID : label
            label_mapping = {
                line.strip().split(',')[0] : line.strip().split(',')[1] 
                for line in in_f
                }
        return label_mapping
    
    def read_embeddings(self,model_dir):
        emb_local_p = model_dir / "cath_v430_dom_seqs_S100_161121.npy"
       
        if not emb_local_p.is_file():
            embeddings_link = "https://rostlab.org/~deepppi/eat_dbs/cath_v430_dom_seqs_S100_161121.npy"
            download_file(embeddings_link,emb_local_p)
        embeddings = torch.from_numpy( np.load(emb_local_p) ).to(device)
        embeddings = self.model.single_pass(embeddings)
        
        id_file_name = "cath_v430_dom_seqs_S100_161121.txt"
        id_local_p = model_dir / id_file_name
        if not id_local_p.is_file():
            weights_link = "https://rostlab.org/~deepppi/eat_dbs/" + id_file_name
            download_file(weights_link,id_local_p)
            
        with open(id_local_p,'r') as in_f:
            # cath|4_3_0|126lA00_1-162 --> 126lA00
            ids = [ line.strip().split("|")[-1].split("_")[0] for line in in_f ]
    
        return ids, embeddings.unsqueeze(dim=0)
    
    def write_predictions(self, predictions, out_dir):
        out_p = out_dir / "prottucker_CATH_pred.csv"
        with open(out_p,"w+") as out_f:
            out_f.write("\n".join( [ lookup_id + "\t" + cath_anno + "\t{:.3f}".format(nn_dist)
                                     for lookup_id, cath_anno, nn_dist in predictions
                                     ]
                                )
                    )
        return None

#### Protein-level predictors
class LA():
    def __init__(self, model_dir, output_dim, use_onnx=False):
        self.subcell=True if output_dim==10 else False
        if use_onnx:
            model_dir = model_dir / "light_attention_onnx"
            if  not model_dir.is_dir():
                print(f"No light attention model available. The onnx model must be at {model_dir}")
            if self.subcell:
                la_subcell_model_path = f"{model_dir}/la_subcell.onnx"
                try:
                    self.model = ort.InferenceSession(la_subcell_model_path)
                except NoSuchFile:
                    print(f"ERROR: No onnx LA subcell model at path {la_subcell_model_path}.")
                    quit()
            else:
                la_model_path = f"{model_dir}/la.onnx"
                try:
                    self.model = ort.InferenceSession(la_model_path)
                except NoSuchFile:
                    print(f"ERROR: No onnx LA model at path {la_model_path}.")
                    quit()

        else:
            model_dir = model_dir / "light_attention"
            if  not model_dir.is_dir():
                model_dir.mkdir()
            self.model = self.load_model(model_dir, output_dim)

    def load_model(self, model_dir, output_dim):
        if self.subcell:
            checkpoint_p = model_dir / "la_prott5_subcellular_localization.pt"
            weights_link = 'http://data.bioembeddings.com/public/embeddings/feature_models/light_attention/la_prott5_subcellular_localization.pt'
        else:
            checkpoint_p = model_dir / "la_prott5_subcellular_solubility.pt"
            weights_link = 'http://data.bioembeddings.com/public/embeddings/feature_models/light_attention/la_prott5_subcellular_solubility.pt'

        model = LightAttention(output_dim=output_dim)
        # LA has a normalization layer --> cast to fp32 for stability
        return load_model(model,weights_link,checkpoint_p).float()

    # https://github.com/sacdallago/bio_embeddings/blob/develop/bio_embeddings/extract/light_attention/light_attention_annotation_extractor.py#L15
    def class2label(self):
        if self.subcell:
            return {
                0: "Cell_membrane",
                1: "Cytoplasm",
                2: "Endoplasmatic_reticulum",
                3: "Golgi_apparatus",
                4: "Lysosome_or_Vacuole",
                5: "Mitochondrion",
                6: "Nucleus",
                7: "Peroxisome",
                8: "Plastid",
                9: "Extracellular"
            }
        else:
            return {
                0: "Membrane",
                1: "Soluble"
            }

    def write_predictions(self,predictions,out_dir):
        class2label=self.class2label()
        if self.subcell:
            out_p = out_dir / "la_subcell_pred.txt"
        else:
            out_p = out_dir / "la_mem_pred.txt"

        with open(out_p, 'w+') as out_f:
            out_f.write( '\n'.join( [ class2label[pred] for pred in predictions] ) )
        return None


#### 3D Structure prediction:
class Ember3D:
        
    def __init__(self, model_dir):
        model_dir = model_dir / "ember3D"
        if not model_dir.is_dir():
            model_dir.mkdir()
            
        self.model, self.Ember3D_Result = self.load_model(model_dir)

    def load_model(self, model_dir):
        # EMBER3D specifics 
        # Import here to avoid unnecessary dependencies if 3D output is not wanted
        import sys
        sys.path.insert(0, './EMBER3D')
        from EMBER3D.model import RF_1I1F
        from EMBER3D.Ember3D import Ember3D_Result
        checkpoint_p = model_dir / "EMBER3D.model"
        weights_link = 'https://github.com/kWeissenow/EMBER3D/raw/main/model/EMBER3D.model'
        
        if not checkpoint_p.exists():
          download_file(weights_link,checkpoint_p)
          
        model = RF_1I1F()
        model = model.to(device)
        model.load_state_dict(torch.load(checkpoint_p))
        model.eval()

        return model, Ember3D_Result

    # original one-hot implementation of Konstantin
    def sequence_to_onehot(self, seq):
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        # replace unknown/ambigious/rare AAs (X) by Glycin for 3D reconstruction (simply assume no side-chain in this case)
        encoded = torch.tensor([aa_list.index(c) if c not in "XOUZB" else aa_list.index("G") for c in seq])
        return torch.nn.functional.one_hot(encoded, num_classes=20)

    def write_predictions(self,predictions,ids,out_dir):
        out_dir = out_dir / "ember3d_pdbs"
        if not out_dir.is_dir():
            out_dir.mkdir()
            
        # write one PDB file for each protein
        for (prot_id, result) in zip(ids,predictions):
            result.save_pdb(prot_id, str(out_dir/(prot_id+".pdb")))
            
        return None


    def predict(self, seq, emb_1d, emb_2d):
        with torch.no_grad():
            emb_1d = torch.unsqueeze(emb_1d, dim=0)
            emb_1d = torch.unsqueeze(emb_1d, dim=0)
            emb_2d = torch.permute(emb_2d, (1, 2, 0))
            emb_2d = torch.unsqueeze(emb_2d, dim=0)

            seq1hot = self.sequence_to_onehot(seq)
            seq1hot = torch.unsqueeze(seq1hot, dim=0).to(device)

            idx = torch.arange(len(seq))
            idx = torch.unsqueeze(idx, dim=0).to(device)
            
            with torch.cuda.amp.autocast(): # actual 3D prediction
                pair_pred, coords_pred, lddt_pred = self.model.forward(seq1hot, idx, emb_1d, emb_2d)
                
            pair_pred = list(pair_pred)
            for i in range(len(pair_pred)):
                pair_pred[i] = pair_pred[i].detach()
            result = self.Ember3D_Result(seq, pair_pred, coords_pred.detach(), lddt_pred.squeeze().detach())

            return result

#### Generators:
# ProtGPT2 will generate new proteins if no FASTA file is passed to the script
# Adapted from: https://huggingface.co/spaces/sdnf/GradioFold/blob/main/app.py#L409
class ProtGPT2():
    def __init__(self,model_dir):
        model_dir = model_dir / "protgpt2"
        if  not model_dir.is_dir():
            model_dir.mkdir()
        # Load ProTGPT2
        self.model, self.tokenizer = self.load_protGPT2(model_dir)
        self.bad_words_ids = self.get_bad_word_ids()
        # the seed used as starting point for generating new random sequences
        self.start_seq = "<|endoftext|>"
        # the token indicating the end of a generated sequence.
        # currently identical to start_sequences but can be different if a different starting_point is known (prior knowledge)
        self.eos = "<|endoftext|>"
        # Necessary for perplexity computation 
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        
    def get_bad_word_ids(self):
        # avoid generating proteins with unknown/ambigious/rare AAs
        kmer_X = [ kmer for kmer in self.tokenizer.get_vocab().keys() 
                  if "X" in kmer or "Z" in kmer or "B" in kmer or "O" in kmer ]
        print(kmer_X)
        bad_words_ids = self.tokenizer(kmer_X, add_special_tokens=False).input_ids
        return bad_words_ids
    
    def load_protGPT2(self,model_dir):
        print("Start loading ProtGPT...")
        start=time.time()
        transformer_name="nferruz/ProtGPT2"
        model = GPT2LMHeadModel.from_pretrained(transformer_name,cache_dir=model_dir)
        tokenizer = GPT2Tokenizer.from_pretrained(transformer_name,cache_dir=model_dir)
        # Artificially add padding token (necessary for batch-processing of perplexity computation)
        tokenizer.pad_token = tokenizer.eos_token
        model = model.half() # TODO: double check with Noelia whether that's ok
        model = model.to(device)
        model = model.eval()
        print("Finished loading: {} in {:.1f}[s]".format(transformer_name,time.time()-start))
        return model, tokenizer
    
    # Mostly taken/modified from: https://huggingface.co/spaces/evaluate-metric/perplexity/blob/main/perplexity.py
    def batch_compute_perplexity(self, seqs, add_start_token=True):
        token_encoding = self.tokenizer.batch_encode_plus(seqs,padding="longest")

        input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
        attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)

        if add_start_token: # add start token to each sequence to mimick training as closely as possible
            bos_tokens_tensor = torch.tensor([[self.tokenizer.bos_token_id]] * input_ids.size(dim=0)).to(device)
            input_ids = torch.cat([bos_tokens_tensor, input_ids], dim=1)
            attention_mask = torch.cat(
                [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attention_mask], dim=1
            )

        with torch.no_grad():
            out_logits = self.model(input_ids, attention_mask=attention_mask).logits

        # generate targets to compute ppl. (in this case: copy of inputs that gets shifted to the right)
        target_ids = input_ids
        #target_ids[~attention_mask] = -100 # mask padded tokens from loss/ppl computation
        
        # for the last token, we can not predict any next token --> remove
        shift_logits = out_logits[..., :-1, :].contiguous()
        # removing the first token from targets effectively shifts everything to the right
        shift_labels = target_ids[..., 1:].contiguous()
        # same goes for attention-mask: removing first element, effectively shifts right
        shift_attention_mask_batch = attention_mask[..., 1:].contiguous()

        perplexity_batch = torch.exp2(
            (self.ce_loss(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
            / shift_attention_mask_batch.sum(1)
        )

        ppls = perplexity_batch.tolist()
        return ppls
    
    
    # parameters directly taken from: https://huggingface.co/spaces/sdnf/GradioFold/blob/main/app.py#L409
    def run_protgpt2(self, n_gen, length=100, repetitionPenalty=1.2, top_k_poolsize=950):
        print("Start generating proteins with ProtGPT2 ...")
        start=time.time()
        n_seqs_to_select = 50 # max. number of sequences to select from each batch
        n_seqs_to_sample = n_seqs_to_select*10 # how many samples to generate per batch; only subset of generated are used (filtering based on PPL)
        seq_dict = dict()
        input_ids = self.tokenizer.encode(self.start_seq, return_tensors='pt').to(device)
        
        while len(seq_dict)<n_gen: # generate new sequences batch-wise until desired size is reached
            
            hallucinated_proteins = self.model.generate(input_ids,
                            max_length=length, 
                            do_sample=True, 
                            top_k=top_k_poolsize,  
                            repetition_penalty=repetitionPenalty, 
                            num_return_sequences=n_seqs_to_sample,
                            eos_token_id=0,
                            pad_token_id=0, # this avoids raising a warning
                            bad_words_ids=self.bad_words_ids
                            ) 

            clean_seqs = list()
            filtered_seqs = list()

            # "clean up" newly generated sequences IF
            # new-lines are generated after few tokens or
            # End-of-sentence is reached too early/often
            for rnd_seq in hallucinated_proteins:
                decoded_seq = self.tokenizer.decode(rnd_seq)
                
                if (decoded_seq[0:60].count("\n")==1 and # No newlines allowed in first line and avoid truncation
                    decoded_seq.count(self.eos)>=2 and # minimally, two EOS tokens (one for start, one for end)
                    decoded_seq.count("X")<10 # avoid sequences with too many unknowns
                    ):
                    
                    # in case multiple sequences were generated in one pass, use only the first one
                    clean_seq = decoded_seq.split(self.eos)[1]
                    if len(clean_seq)<5: # remove sequences that are too short
                        continue
                    clean_seqs.append(clean_seq)

            # get perplexity for batch of proteins
            # TODO: add perplexity threshold here to filter out low-complexity hallucinations
            ppls = self.batch_compute_perplexity(clean_seqs) 
            filtered_seqs += [ (seq, ppl) for seq, ppl in zip(clean_seqs,ppls) ]
            
            filtered_seqs.sort(key = lambda x: x[1])
            # choose only the 10% generated sequences with best perplexity 
            # TODO: Rather in-efficient but can be skipped once we have a proper threshold; see filtering above
            sel_seqs = filtered_seqs[:n_seqs_to_select] 
            
            # dictionary style is required to be compatible with the output of my read_fasta function
            for i, seq in enumerate(sel_seqs,start=len(seq_dict)):
                s = seq[0].replace("\n","")
                seq_len = len(s)
                identifier=">seq{:.0f}, L={:.0f}, ppl={:.3f}".format(i,seq_len,seq[1])
                seq_dict[identifier]  = s
                if len(seq_dict)==n_gen:
                    print("Generated and filtered {} sequences in {:.1f}[s]".format(len(seq_dict),time.time()-start))
                    return seq_dict  # return once desired number of sequences is reached
                
        print("Generated and filtered {} sequences in {:.1f}[s]".format(len(seq_dict),time.time()-start))
        return seq_dict

#### Embedding/Inference
# ProtT5 acts as a microscope with different lenses/predictors to investigate 
# various protein properties
class ProtT5Microscope():
    
    def __init__( self, seq_dict, model_dir, fmt, use_onnx_model=False):
        self.predictors, self.results = self.register_predictors(model_dir,
                                                                 fmt,
                                                                 use_onnx_model=use_onnx_model)
        # only output attention heads if 3D output is requested
        output_attentions=True if "EMBER3D" in self.predictors else False
            
        self.prott5, self.tokenizer = self.get_T5_model(model_dir,output_attentions)
        self.seq_dict = seq_dict

        self.ids = list()
        self.seqs = list()
        self.sigm = nn.Sigmoid()
    
    
    def register_predictors(self, model_dir, fmt, use_onnx_model:bool):
        '''

        Parameters
        ----------
        model_dir : PATH
            Path to the checkpoint directory 
        fmt : LIST
            list of predictors to run
        use_onnx_model : BOOL
            If the model should be loaded from an onnx file

        Returns
        -------
        p : DICT
            dictionary of predictors. 
            Key refers to the name of the predictor. One predictor can have multiple outputs
        r : DICT
            dictionary of results (lists) to store predictions. 
            Some predictors have multiple outputs and need multiple output containers

        '''
        p = {} # dictionary holding predictor classes
        r = {} # dict for storing lists of results results
        # currently supported: ss,cons,dis,mem,bind, go,subcell,tucker,ember,emb
        for f in fmt:
            if f=="ss":
                print("Loading secondary structure predictor")
                p["ProtT5_SecStruct"] = SecStructPredictor(model_dir, use_onnx=use_onnx_model)
                r["SecStruct3"] = list() # 3-state secondary structure
                r["SecStruct8"] = list() # 3-state secondary structure
                
            elif f=="cons":
                print("Loading conservation predictor")
                p["VESPA_Conservation"] = ConservationPredictor(model_dir=model_dir, use_onnx=use_onnx_model)
                r["Conservation"] = list()
                
            elif f=="dis":
                print("Loading disorder predictor")
                p["SETH"] = SETH(model_dir, use_onnnx_model=use_onnx_model)
                r["Disorder"] = list()
                
            elif f=="mem":
                print("Loading transmembrane predictor")
                p["TMbed"] = TMbed(model_dir=model_dir, use_onnx_model=use_onnx_model)
                r["Membrane"] = list()
                
            elif f=="bind":
                print("Loading bindinx predictor")
                p["BindEmbed21DL"] = BindPredict(model_dir, use_onnx_model=use_onnx_model)
                r["Binding"] = list()
                
            elif f=="go":
                print("Loading GO predictor")
                p["goPredSim-mfo"] = GoPredSim(model_dir, "mfo")
                p["goPredSim-bpo"] = GoPredSim(model_dir, "bpo")
                p["goPredSim-cco"] = GoPredSim(model_dir, "cco")
                r["GO-mfo"] = list()
                r["GO-bpo"] = list()
                r["GO-cco"] = list()
                
            elif f=="subcell":
                print("Loading subcellular location predictor")
                p["LA-subcell"] = LA(model_dir,output_dim=10, use_onnx=use_onnx_model)
                p["LA-mem"] = LA(model_dir,output_dim=2, use_onnx=use_onnx_model)
                r["Subcell"] = list()
                r["LA-mem"] = list()
                
            elif f=="tucker":
                print("Loading CATH predictor")
                p["ProtTucker"] = ProtTucker(model_dir)
                r["ProtTucker"] = list()
                
            elif f=="ember3D":
                print("Loading 3D structure predictor")
                p["EMBER3D"] = Ember3D(model_dir)
                r["3DStructure"] = list()
                
            elif f=="emb":
                p["ProtEmbs"] = None # flag to highlight embeddings are requested
                r["ProtEmbs"] = list() # list of per-protein embeddings
                
            else:
                print("The requested predictor is not implemted. Check list of available predictors (-fmt parameter)")
                raise NotImplementedError
        return p, r
    
    def write_list(self,data,out_p):
        with open(out_p,"w+") as out_f:
            out_f.write("\n".join(data))
        return None
    
    def write_protEmbs(self,out_dir):
        out_p = out_dir / "protein_embeddings.npy"
        np.save(str(out_p),self.results["ProtEmbs"])
        return None
    
    def get_T5_model(self,model_dir,output_attentions):
        # Load your checkpoint here
        # Currently, only the encoder-part of ProtT5 is loaded in half-precision
        print("Start loading ProtT5...")
        start=time.time()
        transformer_name = "Rostlab/prot_t5_xl_half_uniref50-enc"
        prott5_dir = model_dir / "prott5"
        model = T5EncoderModel.from_pretrained(transformer_name, torch_dtype=torch.float16, 
                                               cache_dir=prott5_dir, output_attentions=output_attentions)
        model = model.to(device)
        model = model.eval()
        tokenizer = T5Tokenizer.from_pretrained(transformer_name, do_lower_case=False, cache_dir=prott5_dir)
        
        print("Finished loading: {} in {:.1f}[s]".format(transformer_name,time.time()-start))
        return model, tokenizer

    def batch_predict_resedues_from_loaded_embs(self,
                                                path_to_embeddings='Embeddings/Rostlab_prot_t5_xl_uniref50.h5',
                                                max_batch_size=100, max_residues=4000,
                                                use_onnx_model=False):
        embeddings = read_h5_embeddings(file_path=path_to_embeddings)
        sorted_seqs = sorted(embeddings.items(), key=lambda kv: len(embeddings[kv[0]]),
                             reverse=True)
        batch = list()
        start = time.time()
        for seq_idx, (seq_description, embedded_seq) in tqdm(enumerate(sorted_seqs, 1)):
            seq_len = len(embedded_seq)
            batch.append((seq_description, embedded_seq, seq_len))
            n_res_batch = sum([s_len for _, _, s_len in batch]) + seq_len

            # if not enough sequences were accumulated to process one batch continue
            if not (len(batch) >= max_batch_size or  # max. number of sequences per batch
                    n_res_batch >= max_residues or  # max. number of residues per batch
                    seq_idx == len(sorted_seqs)  # special case: last batch
            ):
                continue
            else:
                seq_descriptions, embedded_seqs, seq_lens = zip(*batch)
                batch = list()

                # Manual padding
                max_length = max(array.shape[0] for array in embedded_seqs)
                padded_arrays = []
                attention_masks = []

                for array in embedded_seqs:
                    padding = ((0, max_length - array.shape[0]), (0, 0))  # ((Pad oben, Pad unten), (Pad links, Pad rechts))
                    padded_array = np.pad(array, padding, mode='constant', constant_values=0)
                    attention_mask = np.ones(array.shape[0], dtype=int)
                    pad_mask = np.zeros(max_length - array.shape[0], dtype=int)
                    full_attention_mask = np.concatenate([attention_mask, pad_mask])
                    padded_arrays.append(padded_array)
                    attention_masks.append(full_attention_mask)

                padded_arrays = np.stack(padded_arrays)
                attention_mask_numpy = np.float32(np.stack(attention_masks))
                attention_mask = torch.from_numpy(attention_mask_numpy)
                residue_embedding = torch.from_numpy(np.float32(padded_arrays))
                protein_embeddings = []
                for idx, s_len in enumerate(seq_lens):
                    # get per-protein embeddings here (take padding into account)
                    protein_embeddings.append(residue_embedding[idx,:s_len].mean(dim=0) )
                # stack them again in one tensor for batch-processing
                #protein_embeddings=torch.vstack(protein_embeddings)
                residue_embedding_transpose = torch.permute(residue_embedding, (0,2,1))

                with torch.no_grad():
                    for predictor_name, predictor in self.predictors.items():

                        if predictor_name=="TMbed":
                            # for each registered predictor
                            B,L,_ = residue_embedding.shape
                            # prediction container to gather ensemble predictions
                            # 5 due to an ensemble of 5 models
                            pred = torch.zeros((B, 5, L), device=device,dtype=torch.float32)
                            for model in predictor.model:
                                if use_onnx_model:
                                    ort_inputs = {'input': residue_embedding.numpy(),
                                                  'mask': attention_mask.numpy()}
                                    y = model.run(None, ort_inputs)
                                    y = torch.from_numpy(np.float32(np.stack(y[0])))
                                else:
                                    # TMbed has a normalization layer --> use fp32 for stability
                                    y = model(residue_embedding.float(), attention_mask.float())
                                pred = pred + torch.softmax(y, dim=1)

                            probabilities = (pred / len(predictor.model))
                            mem_Yhat = toCPU( predictor.decoder(probabilities, attention_mask) ).astype(np.byte)
                        elif predictor_name=="VESPA_Conservation":
                            if use_onnx_model:
                                ort_inputs = {predictor.model.get_inputs()[0].name: residue_embedding.numpy()}
                                cons_Yhat = predictor.model.run(None, ort_inputs)
                                cons_Yhat = torch.from_numpy(np.float32(np.stack(cons_Yhat[0])))
                                cons_Yhat = toCPU(torch.max( cons_Yhat, dim=-1, keepdim=True )[1]).astype(np.byte)
                            else:
                                cons_Yhat = predictor.model(residue_embedding)
                                cons_Yhat = toCPU(torch.max( cons_Yhat, dim=-1, keepdim=True )[1]).astype(np.byte)
                        # predict 3- and 8-state sec. struct
                        elif predictor_name=="ProtT5_SecStruct":
                            if use_onnx_model:
                                ort_inputs = {predictor.model.get_inputs()[0].name: residue_embedding.numpy()}
                                d3_Yhat, d8_Yhat = predictor.model.run(None, ort_inputs)
                                d3_Yhat = torch.from_numpy(np.float32(np.stack(d3_Yhat)))
                                d8_Yhat = torch.from_numpy(np.float32(np.stack(d8_Yhat)))

                            else:
                                d3_Yhat, d8_Yhat = predictor.model(residue_embedding)
                            d3_Yhat = toCPU(torch.max( d3_Yhat, dim=-1, keepdim=True )[1] ).astype(np.byte)
                            d8_Yhat = toCPU(torch.max( d8_Yhat, dim=-1, keepdim=True )[1] ).astype(np.byte)
                        elif predictor_name=="BindEmbed21DL":
                            B, L, _ = residue_embedding.shape
                            # container for adding predictions of individual models in the ensemble
                            ensemble_container = torch.zeros( (B, 3, L), device=device,dtype=torch.float16)
                            for model in predictor.model: # for each model in the ensemble
                                if use_onnx_model:
                                    ort_inputs = {model.get_inputs()[0].name: residue_embedding_transpose.numpy()}
                                    model_output_numpy = model.run(None, ort_inputs)
                                    model_output_torch = torch.from_numpy(np.float32(np.stack(model_output_numpy[0])))
                                    pred = self.sigm(model_output_torch)
                                    pred = torch.from_numpy(np.float32(np.stack(pred)))
                                else:
                                    pred = self.sigm( model(residue_embedding_transpose) )
                                ensemble_container = ensemble_container + pred
                            # normalize
                            bind_Yhat = ensemble_container / len(predictor.model)
                            # B x 3 x L --> B x L x 3
                            bind_Yhat = torch.permute(bind_Yhat,(0,2,1))
                            bind_Yhat = toCPU(bind_Yhat>0.5).astype(np.byte)
                        ### Protein-level predictions ###
                        # Light-attention predicts 10 subcellular localizations
                        elif predictor_name=="LA-subcell":
                            if use_onnx_model:
                                ort_inputs = {'input': residue_embedding_transpose.numpy(),
                                              'mask': attention_mask.numpy()}
                                subcell_Yhat = predictor.model.run(None, ort_inputs)
                                subcell_Yhat = torch.from_numpy(np.float32(np.stack(subcell_Yhat[0])))
                            else:
                                # Light attention has a batch-norm layer --> use fp32 for stability
                                subcell_Yhat = predictor.model(residue_embedding_transpose.float(),attention_mask)
                            subcell_Yhat = toCPU(torch.max(subcell_Yhat, dim=1)[1]).astype(np.byte)
                        # Light-attention predicts also membrane-bound vs water-soluble
                        elif predictor_name=="LA-mem":
                            if use_onnx_model:
                                ort_inputs = {'input': residue_embedding_transpose.numpy(),
                                              'mask': attention_mask.numpy()}
                                la_mem_Yhat = predictor.model.run(None, ort_inputs)
                                la_mem_Yhat = torch.from_numpy(np.float32(np.stack(la_mem_Yhat[0])))
                            else:
                                la_mem_Yhat = predictor.model(residue_embedding_transpose.float(),attention_mask)
                            la_mem_Yhat = toCPU(torch.max(la_mem_Yhat, dim=1)[1]).astype(np.byte)
                        elif predictor_name=="SETH":
                            if use_onnx_model:
                                ort_inputs = {predictor.model.get_inputs()[0].name: residue_embedding.numpy()}
                                diso_Yhat = predictor.model.run(None, ort_inputs)
                                diso_Yhat = toCPU(torch.from_numpy(np.float32(np.stack(diso_Yhat[0]))))
                            else:
                                diso_Yhat = toCPU( predictor.model(residue_embedding) )

                for batch_idx, identifier in enumerate(seq_descriptions):
                    s_len = seq_lens[batch_idx] # get sequence length of query
                    self.ids.append(identifier ) # store IDs
                    self.seqs.append(embedded_seqs[batch_idx]) # store sequence
                    for predictor_name, predictor in self.predictors.items():
                        if predictor_name=="TMbed":
                            self.results["Membrane"].append(mem_Yhat[batch_idx,:s_len])
                        elif predictor_name=="VESPA_Conservation":
                            self.results["Conservation"].append(cons_Yhat[batch_idx,:s_len])
                        elif predictor_name=="ProtT5_SecStruct":
                            self.results["SecStruct3"].append(d3_Yhat[batch_idx,:s_len])
                            self.results["SecStruct8"].append(d8_Yhat[batch_idx,:s_len])
                        elif predictor_name=="BindEmbed21DL":
                            self.results["Binding"].append(bind_Yhat[batch_idx,:s_len,:])
                        elif predictor_name=="LA-subcell":
                            self.results["Subcell"].append(subcell_Yhat[batch_idx])
                        elif predictor_name=="LA-mem":
                            self.results["LA-mem"].append(la_mem_Yhat[batch_idx])
                        elif predictor_name=="SETH":
                            self.results["Disorder"].append(diso_Yhat[batch_idx,:s_len])
        exe_time = time.time()-start
        print('Total time for generating embeddings and gathering predictions: ' +
              '{:.2f} [s] ### Avg. time per protein: {:.3f} [s]'.format(
                  exe_time, exe_time/len(self.ids) ))
        return None

    def batch_predict_residues( self, max_batch_size=100, max_residues=4000):
        sorted_seqs = sorted( self.seq_dict.items(), key=lambda kv: len( self.seq_dict[kv[0]] ), reverse=True )
        batch = list()
        print("Start predicting protein properties ...")
        start = time.time()
        
        # for each sequence in the batch
        for seq_idx, (pdb_id, seq) in tqdm(enumerate(sorted_seqs,1)):
            seq_len = len(seq)
            seq = ' '.join(list(seq))
            
            batch.append((pdb_id,seq,seq_len))
            
            n_res_batch = sum([ s_len for  _, _, s_len in batch ]) + seq_len 
            # if enough sequences were accumulated to process one batch
            if ( len(batch)>=max_batch_size or  # max. number of sequences per batch
                    n_res_batch>=max_residues or # max. number of residues per batch
                    seq_idx==len(sorted_seqs) # special case: last batch
                    ):
                
                pdb_ids, seqs, seq_lens = zip(*batch)
                batch = list()
                
                # tokenize protein sequence and generate attention_mask for batch_processing
                token_encoding = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding='longest')
                input_ids      = torch.tensor(token_encoding['input_ids']).to(device)
                attention_mask = torch.tensor(token_encoding['attention_mask']).to(device)
                
                try:
                    with torch.no_grad():
                        prott5_output = self.prott5( input_ids, attention_mask=attention_mask )
                except RuntimeError:
                    print("RuntimeError for {} (L={})".format(pdb_id, seq_lens))
                    print("Cleaning up ...")
                    del(token_encoding)
                    del(input_ids)
                    del(attention_mask)
                    continue
                
                # extract last hidden states (=embeddings)
                residue_embedding = prott5_output.last_hidden_state.detach()
                # mask out padded elements in the attention output (can be non-zero) for further processing/prediction
                residue_embedding = residue_embedding*attention_mask.unsqueeze(dim=-1)
                # cast to fp32 (important to ascertain performance of downstream predictors, i.e., TMbed's layerNorm)
                residue_embedding = residue_embedding
                
                protein_embeddings = list()
                # ProtT5 appends a special tokens at the end of each sequence
                # Mask this also out during inference
                for idx, s_len in enumerate(seq_lens):
                    attention_mask[idx,s_len] = 0
                    # get per-protein embeddings here (take padding into account)
                    protein_embeddings.append( residue_embedding[idx,:s_len].mean(dim=0) )
                # stack them again in one tensor for batch-processing
                protein_embeddings=torch.vstack(protein_embeddings)
                
                # some predictors require dimensions to be swapped; prepare only once here
                # B x L x N --> B x N x L
                residue_embedding_transpose = torch.permute(residue_embedding, (0,2,1))
                with torch.no_grad():
                    
                    # batch-predict all properties but 3D structure
                    # 3D structure predictions works only without batching right now
                    for predictor_name, predictor in self.predictors.items(): # for each registered predictor
                        
                        ### Residue-level predictions ###
                        # predict 3- and 8-state sec. struct
                        if predictor_name=="ProtT5_SecStruct": 
                            d3_Yhat, d8_Yhat = predictor.model(residue_embedding)
                            d3_Yhat = toCPU(torch.max( d3_Yhat, dim=-1, keepdim=True )[1] ).astype(np.byte)
                            d8_Yhat = toCPU(torch.max( d8_Yhat, dim=-1, keepdim=True )[1] ).astype(np.byte)
                            
                        elif predictor_name=="VESPA_Conservation":
                            cons_Yhat = predictor.model(residue_embedding)
                            cons_Yhat = toCPU(torch.max( cons_Yhat, dim=-1, keepdim=True )[1]).astype(np.byte)
                            
                        elif predictor_name=="SETH":
                            diso_Yhat = toCPU( predictor.model(residue_embedding) )
                            
                        elif predictor_name=="TMbed":
                            B,L,_ = residue_embedding.shape
                            # prediction container to gather ensemble predictions
                            # 5 due to an ensemble of 5 models
                            pred = torch.zeros((B, 5, L), device=device,dtype=torch.float32)
                            
                            for model in predictor.model:
                                # TMbed has a normalization layer --> use fp32 for stability
                                y = model(residue_embedding.float(), attention_mask.float())
                                pred = pred + torch.softmax(y, dim=1)
                                
                            probabilities = (pred / len(predictor.model))
                            mem_Yhat = toCPU( predictor.decoder(probabilities, attention_mask) ).astype(np.byte)
                        
                        elif predictor_name=="BindEmbed21DL":
                            B, L, _ = residue_embedding.shape
                            # container for adding predictions of individual models in the ensemble
                            ensemble_container = torch.zeros( (B, 3, L), device=device,dtype=torch.float16)
                            for model in predictor.model: # for each model in the ensemble
                                pred = self.sigm( model(residue_embedding_transpose) )
                                ensemble_container = ensemble_container + pred
                            # normalize
                            bind_Yhat = ensemble_container / len(predictor.model)
                            # B x 3 x L --> B x L x 3
                            bind_Yhat = torch.permute(bind_Yhat,(0,2,1))
                            bind_Yhat = toCPU(bind_Yhat>0.5).astype(np.byte)
                            
                        ### Protein-level predictions ###
                        # Light-attention predicts 10 subcellular localizations
                        elif predictor_name=="LA-subcell":
                            # Light attention has a batch-norm layer --> use fp32 for stability
                            subcell_Yhat = predictor.model(residue_embedding_transpose.float(),attention_mask)
                            subcell_Yhat = toCPU(torch.max(subcell_Yhat, dim=1)[1]).astype(np.byte)
                        # Light-attention predicts also membrane-bound vs water-soluble
                        elif predictor_name=="LA-mem":
                            la_mem_Yhat = predictor.model(residue_embedding_transpose.float(),attention_mask)
                            la_mem_Yhat = toCPU(torch.max(la_mem_Yhat, dim=1)[1]).astype(np.byte)
                        ### EAT-based methods ###
                        elif predictor_name=="ProtTucker":
                             tuckered_embeddings = predictor.model.single_pass(protein_embeddings)
                             prottucker_Yhat = eat(predictor.lookup_embs, 
                                              predictor.lookup_ids, 
                                              predictor.lookup_labels, 
                                              tuckered_embeddings, 
                                              threshold=None # apply no threshold; if threshold wanted, use 1.1
                                              )

                        elif predictor_name=="goPredSim-mfo":
                            go_mfo_Yhat = eat(predictor.lookup_embs, 
                                              predictor.lookup_ids, 
                                              predictor.lookup_labels, 
                                              protein_embeddings, 
                                              threshold=None # apply no threshold; if threshold wanted, use 0.5 
                                              )
                        elif predictor_name=="goPredSim-bpo":
                            go_bpo_Yhat = eat(predictor.lookup_embs, 
                                              predictor.lookup_ids, 
                                              predictor.lookup_labels, 
                                              protein_embeddings, 
                                              threshold=None # apply no threshold; if threshold wanted, use 0.5 
                                              )
                        elif predictor_name=="goPredSim-cco":
                            go_cco_Yhat = eat(predictor.lookup_embs, 
                                              predictor.lookup_ids, 
                                              predictor.lookup_labels, 
                                              protein_embeddings, 
                                              threshold=None # apply no threshold; if threshold wanted, use 0.5 
                                              )
                            
                        
                for batch_idx, identifier in enumerate(pdb_ids):
                    s_len = seq_lens[batch_idx] # get sequence length of query
                    self.ids.append(identifier ) # store IDs
                    seq = "".join(seqs[batch_idx].split()) 
                    self.seqs.append(seq) # store sequence
                    
                    for predictor_name, predictor in self.predictors.items():
                        if predictor_name=="ProtT5_SecStruct":
                            self.results["SecStruct3"].append(d3_Yhat[batch_idx,:s_len])
                            self.results["SecStruct8"].append(d8_Yhat[batch_idx,:s_len])
                        elif predictor_name=="VESPA_Conservation":
                            self.results["Conservation"].append(cons_Yhat[batch_idx,:s_len])
                        elif predictor_name=="SETH":
                            self.results["Disorder"].append(diso_Yhat[batch_idx,:s_len])
                        elif predictor_name=="TMbed":
                            self.results["Membrane"].append(mem_Yhat[batch_idx,:s_len])
                        elif predictor_name=="LA-subcell":
                            self.results["Subcell"].append(subcell_Yhat[batch_idx])
                        elif predictor_name=="LA-mem":
                            self.results["LA-mem"].append(la_mem_Yhat[batch_idx])
                        elif predictor_name=="ProtTucker":
                            self.results["ProtTucker"].append(prottucker_Yhat[batch_idx])
                        elif predictor_name=="goPredSim-mfo":
                            self.results["GO-mfo"].append(go_mfo_Yhat[batch_idx])
                        elif predictor_name=="goPredSim-bpo":
                            self.results["GO-bpo"].append(go_bpo_Yhat[batch_idx])
                        elif predictor_name=="goPredSim-cco":
                            self.results["GO-cco"].append(go_cco_Yhat[batch_idx])
                        elif predictor_name=="BindEmbed21DL":
                            self.results["Binding"].append(bind_Yhat[batch_idx,:s_len,:])
                        elif predictor_name=="ProtEmbs":
                            self.results["ProtEmbs"].append( protein_embeddings[batch_idx].detach().cpu().numpy().squeeze() )
                        # 3D structure prediction is handled differently as it does not support batch-prediction (yet)
                        elif predictor_name=="EMBER3D":
                            # just to be absolutely sure; probably not necessary
                            with torch.no_grad(): 
                                # prepare input
                                emb_1d = residue_embedding[batch_idx, :s_len]
                                # Tuple of torch.FloatTensor (one for each layer) of shape
                                # (batch_size, num_heads, sequence_length, sequence_length)
                                # (24 x 32 x L x L) 24=n_layer; 32=n_heads
                                emb_2d = torch.cat( [ attention[batch_idx:batch_idx+1,:,:s_len,:s_len] 
                                                     for attention in prott5_output[1]], dim=0).detach()
                                emb_2d = torch.reshape(emb_2d,(768,s_len,s_len))
                                #  symmetrize
                                emb_2d = 0.5 * (emb_2d + torch.transpose(emb_2d, 1, 2))
                                # predict
                                self.results["3DStructure"].append( predictor.predict(seq,emb_1d,emb_2d) )
                            
        exe_time = time.time()-start
        print('Total time for generating embeddings and gathering predictions: ' +
              '{:.2f} [s] ### Avg. time per protein: {:.3f} [s]'.format(
                    exe_time, exe_time/len(self.ids) ))
        return None

#### Utility functions
# predict via embedding-based annotation transfer (EAT)
def eat(lookup_embs, lookup_ids, lookup_labels, queries,threshold, norm=2):
    
    # pairwise distance between lookup set and current batch
    pdist = torch.cdist(lookup_embs, queries.unsqueeze(dim=0), p=norm).squeeze(dim=0)

    # get closes neighbor for each query
    nn_dists, nn_idxs = torch.topk(pdist, 1, largest=False, dim=0)

    predictions=list()
    n_test = queries.shape[0]
    for test_idx in range(n_test):  # for all test proteins
        # index of nearest neighbour (nn) in train set
        nn_idx = int( nn_idxs[:, test_idx] )
        nn_dist = float( nn_dists[:, test_idx] )

        # get id of nn (infer annotation)
        lookup_id = lookup_ids[nn_idx]

        # if a threshold is passed, skip all proteins above this threshold
        if threshold is not None and nn_dist > threshold:
            lookup_id = "N/A"
            lookup_label = "N/A"
        else:
            lookup_label = lookup_labels[lookup_id]
        predictions.append((lookup_id, lookup_label, nn_dist))
    return predictions


def download_file(url,local_path):
    if not local_path.parent.is_dir():
        local_path.parent.mkdir()
        
    print("Downloading: {}".format(url))
    req = request.Request(url, headers={
          'User-Agent' : 'Mozilla/5.0 (Windows NT 6.1; Win64; x64)'
      })
  
    with request.urlopen(req) as response, open(local_path, 'wb') as outfile:
          shutil.copyfileobj(response, outfile)
    return None

    
def load_model(model, weights_link, checkpoint_p,state_dict="state_dict"):
      # if no pre-trained model is available, yet --> download it
      if not checkpoint_p.exists():
          download_file(weights_link,checkpoint_p)

      # Torch load will map back to device from state, which often is GPU:0.
      # to overcome, need to explicitly map to active device
      global device
      if not device:
          device = get_device()
      state = torch.load(checkpoint_p, map_location=device)

      model.load_state_dict(state[state_dict])

      model = model.eval()
      #model = model.half()
      model = model.to(device)

      return model

def read_fasta( fasta_path, split_char="!", id_field=0 ):
    '''
        Reads in fasta file containing multiple sequences.
        Returns dictionary of holding multiple sequences or only single 
        sequence, depending on input file.
        split_char defines character to use for splitting the header
        id_field defines the field to grep the ID from after splitting based on split_char
    '''
    
    seq_dict = dict()
    with open( fasta_path, 'r' ) as fasta_f:
        for line in fasta_f:
            line=line.strip()
            # get uniprot ID from header and create new entry
            if line.startswith('>'):
                uniprot_id = line.split(split_char)[id_field]
                seq_dict[ uniprot_id ] = ''
            else:
                # repl. all whie-space chars and join seqs spanning multiple lines
                # drop gaps and cast to upper-case
                # replace non-standard AAs by "X" (unknown)
                seq = ''.join( line.split() ).upper().replace("-","").replace('U','X').replace('Z','X').replace('O','X')
                seq_dict[ uniprot_id ] +=  seq 
    print("Read in {} proteins.".format(len(seq_dict)))
    return seq_dict

def toCPU(tensor):
    if len(tensor.shape) > 1:
        return tensor.detach().cpu().squeeze(dim=-1).numpy()
    else:
        return tensor.detach().cpu().numpy()
    
def create_arg_parser():
    """"Creates and returns the ArgumentParser object."""

    # Instantiate the parser
    parser = argparse.ArgumentParser(description=( 
            't5_embedder.py creates T5 embeddings for a given text '+
            ' file containing sequence(s) in FASTA-format.') )
    
    # Required positional argument
    parser.add_argument( '-i', '--input', required=False, type=str, default=None,
                    help='A path to a fasta-formatted text file containing protein sequence(s).')

    # Optional positional argument
    parser.add_argument( '-o', '--output', required=True, type=str, 
                    help='A directory for storing results.')

    # Optional positional argument
    parser.add_argument( '-n', '--n_gen', required=False, type=int, default=100, 
                    help='The number of sequences to generate. Only works if no FASTA is passed.')

    # Optional positional argument
    parser.add_argument( '-d', '--device', required=False, type=str, default="", 
                    help='The device to use, e.g. "cuda:2"')

    # Optional positional argument
    parser.add_argument( '-b', '--batch_size', required=False, type=int, default=100, 
                    help='The number of sequences in a ProtT5 batch.')

    # Optional positional argument
    parser.add_argument( '-r', '--residues_per_batch', required=False, type=int, default=4000,
                    help='The max number of residues in a ProtT5 batch.')
    
    # Optional positional argument
    parser.add_argument( '-f', '--fmt', required=False, type=str, default="ss,cons,dis,mem,bind,go,subcell,tucker", 
                    help="""
                    The output format (defines which predictors are run). Defined as comma-separated list.
                    Options are:
                        
                        Predictions made for each residue in a protein:
                        - ss: secondary structure prediction in 3- and 8-states as defined by DSSP (ProtTrans paper)
                        - cons: conservation prediction in 9 classes as defined by ConSeq [0=variable,8=conserved] (VESPA paper)
                        - dis: continuous disorder prediction in the form of CheZOD scores [-14=disorder, 15=order] (SETH paper)
                        - mem: prediction of signal peptides, membrane segments and topology (TMbed paper)
                        - bind: binding residue prediction for small-, metal, nucleic-acid-binding
                        
                        Predictions made for the whole protein (no residue-level annotation but global feature):
                        - go: Gene Ontology (GO) term prediction via embedding-based annotation transfer (EAT) (goPredSim paper)
                        - subcell: Subcell. loc. prediction and binary classification of membrane-bound vs. soluble proteins (Light-Attention paper)
                        - tucker: prediction of protein fold according to CATH via EAT on ProtTucker(ProtT5) embeddings (ProtTucker)
                        
                        Embeddings:
                        - emb: per-protein ProtT5 embeddings (mean-pooled over length dimension).
                        
                        3D structure:
                        - ember3D: generate backbone 3D coordinates using EMBER3D
                        
                    default=ss,cons,dis,mem,bind,go,subcell,tucker
                    
                    If you want to generate embeddings as well:
                        ss,cons,dis,mem,bind,go,subcell,tucker,emb
                    
                    If you want to generate embeddings as well:
                        ss,cons,dis,mem,bind,go,subcell,tucker,ember,emb
                    """
                    )
    parser.add_argument( '-e', '--embeddings_from_file', required=False, type=str, default='',
                         help='Path to a .h5-File where ProtT5 Embeddings are stored. Leave empty if you want to calculate new embeddings')
    parser.add_argument( '-x', '--onnx', required=False,action="store_true",
                         help='If the model should be loaded from an onnx file.')
    return parser


def main():
    parser = create_arg_parser()
    args   = parser.parse_args()
    
    # Directory for storing all model checkpoints
    root_dir = Path.cwd()
    model_dir = root_dir / "checkpoints"
    if not model_dir.is_dir():
        model_dir.mkdir()
    
    global device

    if args.device != "":
        print("Was using device: {}".format(device))
        device = get_device(args.device)
        print("Now using device: {}".format(device))
    else:
        device = get_device()
        print("Using device: {}".format(device))


    # if a input fasta is passed: read it in
    if args.input is not None:
        seq_path = Path( args.input )
        seq_dict = read_fasta(seq_path)
    else: # if no fasta is passed as input: generate random proteins with ProtGPT2
        n_gen = int( args.n_gen )
        protGPT2 = ProtGPT2(model_dir)
        seq_dict = protGPT2.run_protgpt2(n_gen)

    fmt = args.fmt.split(",")
    
    # here your predictions will be stored
    out_dir  = Path( args.output)
    if not out_dir.is_dir():
        out_dir.mkdir()

    # Load model(s) and generate predictions
    microscope=ProtT5Microscope(seq_dict,model_dir,fmt, use_onnx_model=args.onnx)
    if args.embeddings_from_file:
        microscope.batch_predict_resedues_from_loaded_embs(path_to_embeddings=args.embeddings_from_file,
                                                           max_batch_size=args.batch_size,
                                                           max_residues=args.residues_per_batch,
                                                           use_onnx_model=args.onnx)
    else:
        microscope.batch_predict_residues(max_batch_size=args.batch_size,
                                          max_residues=args.residues_per_batch)
        # write sequences
        seq_out_p = out_dir / "seqs.txt"
        microscope.write_list(microscope.seqs, seq_out_p)

    # write IDs
    id_out_p = out_dir / "ids.txt"
    microscope.write_list(microscope.ids,id_out_p)
    
    # write sequences
    if not args.embeddings_from_file:
        seq_out_p = out_dir / "seqs.txt"
        microscope.write_list(microscope.seqs, seq_out_p)


    for f in fmt:
        if f=="ss":
            # write 3-state DSSP
            microscope.predictors["ProtT5_SecStruct"].write_predictions(
                microscope.results["SecStruct3"], out_dir)
            # write 8-state DSSP
            microscope.predictors["ProtT5_SecStruct"].write_predictions(
                microscope.results["SecStruct8"], out_dir, dssp3=False)
            
        elif f=="cons":
            # write conservation prediction
            microscope.predictors["VESPA_Conservation"].write_predictions(
                microscope.results["Conservation"], out_dir)
            
        elif f=="dis":
            microscope.predictors["SETH"].write_predictions(
                microscope.results["Disorder"], out_dir)
            
        elif f=="mem":
            microscope.predictors["TMbed"].write_predictions(
                microscope.results["Membrane"], out_dir)
            
        elif f=="bind":
            microscope.predictors["BindEmbed21DL"].write_predictions(
                microscope.results["Binding"], out_dir)
            
        elif f=="go":
            microscope.predictors["goPredSim-mfo"].write_predictions(
                microscope.results["GO-mfo"], out_dir)
            microscope.predictors["goPredSim-bpo"].write_predictions(
                microscope.results["GO-bpo"], out_dir)
            microscope.predictors["goPredSim-cco"].write_predictions(
                microscope.results["GO-cco"], out_dir)
            
        elif f=="subcell":
            microscope.predictors["LA-subcell"].write_predictions(
                microscope.results["Subcell"], out_dir)
            microscope.predictors["LA-mem"].write_predictions(
                microscope.results["LA-mem"], out_dir)
            
        elif f=="tucker":
             microscope.predictors["ProtTucker"].write_predictions(
                microscope.results["ProtTucker"], out_dir)
             
        elif f=="ember3D":
            microscope.predictors["EMBER3D"].write_predictions(
                microscope.results["3DStructure"], 
                microscope.ids,
                out_dir
                )
        elif f=="emb":
            # write protein embeddings
            microscope.write_protEmbs(out_dir)


    
if __name__=='__main__':
    main()
    
