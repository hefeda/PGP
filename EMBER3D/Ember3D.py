import torch
import numpy as np
import os
import datetime
from Bio import SeqIO
from argparse import ArgumentParser
import matplotlib.pyplot as plt
from T5Embedder import T5Embedder
from model import *


class Ember3D_Result:
    def __init__(self, seq, pair_pred, coords_pred, lddt_pred):
        self.seq = seq
        self.length = len(seq)
        self.pair_pred = pair_pred
        self.coords_pred = coords_pred
        self.lddt_pred = lddt_pred

    def to_pdb(self, id):
        one_to_three = {
            "R": "ARG",
            "H": "HIS",
            "K": "LYS",
            "D": "ASP",
            "E": "GLU",
            "S": "SER",
            "T": "THR",
            "N": "ASN",
            "Q": "GLN",
            "C": "CYS",
            "G": "GLY",
            "P": "PRO",
            "A": "ALA",
            "V": "VAL",
            "I": "ILE",
            "L": "LEU",
            "M": "MET",
            "F": "PHE",
            "Y": "TYR",
            "W": "TRP"
        }

        lddt = (self.lddt_pred * 100.0).cpu().numpy()
        coords = self.coords_pred.squeeze().cpu().numpy()

        line = "{:6s}{:5d} {:^4s}{:1s}{:3s} {:1s}{:4d}{:1s}   {:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}{:2s}\n"
        pdb_out = ""

        if id is not None:
            pdb_out += "REMARK {}\n".format(id)

        counter = 1
        for seqpos in range(self.length):
             pdb_out += line.format("ATOM", counter, "N", "", one_to_three[self.seq[seqpos]], "A", seqpos + 1, "", coords[seqpos,0,0], coords[seqpos,0,1], coords[seqpos,0,2], 1, lddt[seqpos], "N", "")
             counter += 1
             pdb_out += line.format("ATOM", counter, "CA", "", one_to_three[self.seq[seqpos]], "A", seqpos + 1, "", coords[seqpos, 1, 0], coords[seqpos, 1, 1], coords[seqpos, 1, 2], 1, lddt[seqpos], "C", "")
             counter += 1
             pdb_out += line.format("ATOM", counter, "C", "", one_to_three[self.seq[seqpos]], "A", seqpos + 1, "", coords[seqpos, 2, 0], coords[seqpos, 2, 1], coords[seqpos, 2, 2], 1, lddt[seqpos], "C", "")
             counter += 1
             pdb_out += line.format("ATOM", counter, "O", "", one_to_three[self.seq[seqpos]], "A", seqpos + 1, "", coords[seqpos, 3, 0], coords[seqpos, 3, 1], coords[seqpos, 3, 2], 1, lddt[seqpos], "O", "")
             counter += 1
        pdb_out += "TER\n"

        return pdb_out

    def save_2d_output(self, filename):
        dist_orig = torch.nn.functional.softmax(self.pair_pred[0], dim=1).reshape(-1, self.length, self.length).permute(1,2,0).cpu().numpy()
        dist = np.zeros((self.length, self.length, 37), dtype=np.float32)
        dist[:, :, 0:36] = dist_orig[:, :, 0:36]
        dist[:, :, 36] = np.sum(dist_orig[:, :, 36:], axis=2)
        omega = torch.nn.functional.softmax(self.pair_pred[1], dim=1).reshape(-1, self.length, self.length).permute(1,2,0).cpu().numpy()
        theta = torch.nn.functional.softmax(self.pair_pred[2], dim=1).reshape(-1, self.length, self.length).permute(1,2,0).cpu().numpy()
        phi = torch.nn.functional.softmax(self.pair_pred[3], dim=1).reshape(-1, self.length, self.length).permute(1,2,0).cpu().numpy()

        np.savez_compressed(filename,
                            dist=dist.astype(np.float16),
                            omega=omega.astype(np.float16),
                            theta=theta.astype(np.float16),
                            phi=phi.astype(np.float16))

    def save_pdb(self, id, filename):
        pdb_out = self.to_pdb(id)

        with open(filename, "w") as f:
            f.write(pdb_out)

    def save_contact_map(self, filename):
        distogram = torch.nn.functional.softmax(self.pair_pred[0], dim=1).squeeze().cpu().numpy()
        contacts = np.sum(distogram[:13, :, :], axis=0)
        plt.imsave(filename, contacts, cmap='hot')

    def save_distance_map(self, filename):
        distance_map = self.get_distance_map()
        plt.imsave(filename, distance_map, cmap='hot_r')

    def get_distance_map(self):
        distogram = torch.nn.functional.softmax(self.pair_pred[0], dim=1).squeeze().cpu().numpy()

        mul = np.swapaxes(np.tile(np.arange(42), (self.length, self.length, 1)), 0, 2)
        distance_classes = (np.sum(distogram * mul, axis=0)).astype(np.int8)

        distance_map = distance_classes * 0.5 + 1.75
        np.fill_diagonal(distance_map, 0.0)

        return distance_map


class Ember3D:
    def __init__(self, model_checkpoint, t5_dir, device):
        self.model = RF_1I1F()
        self.model = self.model.to(device)
        self.model.load_state_dict(torch.load(model_checkpoint))
        self.model.eval()

        self.embedder = T5Embedder(t5_dir, device)
        self.device = device

    def sequence_to_onehot(self, seq):
        aa_list = list("ACDEFGHIKLMNPQRSTVWY")
        encoded = torch.tensor([aa_list.index(c) for c in seq])
        return torch.nn.functional.one_hot(encoded, num_classes=20)

    def predict(self, seq):
        with torch.no_grad():
            emb_1d, emb_2d = self.embedder.get_embeddings(seq)
            emb_1d = torch.unsqueeze(emb_1d, dim=0)
            emb_1d = torch.unsqueeze(emb_1d, dim=0)
            emb_2d = torch.permute(emb_2d, (1, 2, 0))
            emb_2d = torch.unsqueeze(emb_2d, dim=0)

            seq1hot = self.sequence_to_onehot(seq)
            seq1hot = torch.unsqueeze(seq1hot, dim=0).to(self.device)

            idx = torch.arange(len(seq))
            idx = torch.unsqueeze(idx, dim=0).to(self.device)

            pair_pred, coords_pred, lddt_pred = self.model.forward(seq1hot, idx, emb_1d, emb_2d)
            pair_pred = list(pair_pred)
            for i in range(len(pair_pred)):
                pair_pred[i] = pair_pred[i].detach()
            result = Ember3D_Result(seq, pair_pred, coords_pred.detach(), lddt_pred.squeeze().detach())

            return result
