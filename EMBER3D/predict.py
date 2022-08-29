import os
from Bio import SeqIO
from argparse import ArgumentParser
from Ember3D import *

parser = ArgumentParser()
parser.add_argument('-i', '--input', dest="input", type=str, required=True)
parser.add_argument('-o', '--output_dir', dest='output_dir', type=str, required=True)
parser.add_argument('-d', '--device', default='cuda:0', dest="device", type=str)
parser.add_argument('--output-2d', dest="output_2d", action="store_true")
parser.add_argument('--no-pdb', dest="no_pdb", action="store_true")
parser.add_argument('--no-distance-map', dest="no_distance_map", action="store_true")
parser.add_argument('-m', '--model', default="model/EMBER3D.model", dest='model_checkpoint', type=str)
parser.add_argument('--t5-dir', dest='t5_dir', default="./ProtT5-XL-U50/", type=str)
args = parser.parse_args()

# Output directories
pdb_dir = os.path.join(args.output_dir, "pdb")
image_dir = os.path.join(args.output_dir, "images")
dist_orient_dir = os.path.join(args.output_dir, "output_2d")
if not os.path.isdir(args.output_dir):
    os.makedirs(args.output_dir)
if not args.no_pdb and not os.path.isdir(pdb_dir):
    os.makedirs(pdb_dir)
if not args.no_distance_map and not os.path.isdir(image_dir):
    os.makedirs(image_dir)
if args.output_2d and not os.path.isdir(dist_orient_dir):
    os.makedirs(dist_orient_dir)

# Prediction
Ember3D = Ember3D(args.model_checkpoint, args.t5_dir, args.device)

for i,record in enumerate(SeqIO.parse(args.input, "fasta")):
    id = record.id
    seq = str(record.seq)

    if "X" in seq:
        print("Skipping {} because of unknown residues".format(id))
        continue

    print("{}\t{}".format(i, id))

    with torch.cuda.amp.autocast():
        result = Ember3D.predict(seq)

        if args.output_2d:
            result.save_2d_output("{}/{}.npz".format(dist_orient_dir, id))

        if not args.no_pdb:
            result.save_pdb(id, "{}/{}.pdb".format(pdb_dir, id))

        if not args.no_distance_map:
            result.save_distance_map("{}/{}.png".format(image_dir, id))
