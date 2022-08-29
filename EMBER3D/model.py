import torch
import torch.nn as nn
from Attention_module_w_str import InitStr_Network, IterBlock, IterBlock_w_Str, FinalBlock, Pair2Pair
from DistancePredictor import DistanceNetwork
from Transformer import LayerNorm, _get_clones


class RF_1I1F(nn.Module):
    def __init__(self, n_layer=1, d_msa=64, d_pair=128, d_hidden=64, p_drop=0.0, n_head_msa=8, n_head_pair=8, r_ff=4,
                 n_resblock=1, performer_L_opts={"nb_features": 64}, performer_N_opts={"nb_features": 64}):
        super(RF_1I1F, self).__init__()
        SE3_param = {
            "num_layers": 2,
            "num_channels": 16,
            "num_degrees": 2,
            "l0_in_features": 32,
            "l0_out_features": 8,
            "l1_in_features": 4,
            "l1_out_features": 4,
            "num_edge_features": 32,
            "div": 2,
            "num_heads": 4
        }

        self.proj_node = nn.Linear(1024, d_msa)
        self.proj_edge = nn.Linear(768, d_pair) # 100

        self.iter_block_2d = IterBlock(n_layer=n_layer,
                                       d_msa=d_msa, d_pair=d_pair,
                                       n_head_msa=n_head_msa,
                                       n_head_pair=n_head_pair,
                                       r_ff=r_ff,
                                       n_resblock=n_resblock,
                                       p_drop=p_drop,
                                       performer_N_opts=performer_N_opts,
                                       performer_L_opts=performer_L_opts
                                       )

        self.init_str = InitStr_Network(node_dim_in=d_msa, node_dim_hidden=d_hidden,
                                        edge_dim_in=d_pair, edge_dim_hidden=d_hidden,
                                        nheads=4, nblocks=3, dropout=p_drop)

        self.iter_block = IterBlock_w_Str(n_layer=n_layer,
                                          d_msa=d_msa, d_pair=d_pair,
                                          n_head_msa=n_head_msa,
                                          n_head_pair=n_head_pair,
                                          r_ff=r_ff,
                                          n_resblock=n_resblock,
                                          p_drop=p_drop,
                                          performer_N_opts=performer_N_opts,
                                          performer_L_opts=performer_L_opts,
                                          SE3_param=SE3_param
                                          )

        self.final = FinalBlock(n_layer=n_layer, d_msa=d_msa, d_pair=d_pair,
                                n_head_msa=n_head_msa, n_head_pair=n_head_pair, r_ff=r_ff,
                                n_resblock=n_resblock, p_drop=p_drop,
                                performer_L_opts=performer_L_opts, performer_N_opts=performer_N_opts,
                                SE3_param=SE3_param)

        self.c6d_predictor = DistanceNetwork(d_pair, p_drop=p_drop)

    def forward(self, seq1hot, idx, emb_1d, emb_2d):
        node = self.proj_node(emb_1d)
        pair = self.proj_edge(emb_2d)

        node, pair = self.iter_block_2d(node, pair)

        xyz = self.init_str(seq1hot, idx, node, pair)

        node, pair, xyz = self.iter_block(node, pair, xyz, seq1hot, idx, top_k=128)

        node, pair, xyz, lddt = self.final(node, pair, xyz, seq1hot, idx)

        pair = self.c6d_predictor(pair)

        return pair, xyz, lddt
