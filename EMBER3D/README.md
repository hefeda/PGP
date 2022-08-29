# EMBER3D
![EMBER](https://rostlab.org/~conpred/EMBER_sketch_small.png "EMBER")

*Konstantin Weissenow, Michael Heinzinger, Burkhard Rost*

*Technical University of Munich*

This repository contains the code for the EMBER3D protein structure and mutation effect prediction system. EMBER3D is currently provided as a **prototype release** for preview purposes. The system is still under active development.

## Installation

Create a new virtual environment, e.g. using conda:
```
conda create -n EMBER3D python=3.8
conda activate EMBER3D
```

If you use CUDA 11, you can use the provided requirements.txt to install dependencies:
```
pip install -r requirements.txt
```

If you plan to render protein mutation movies, you additionally need PyMOL and ffmpeg, which you can install with e.g.
```
conda install -c conda-forge ffmpeg
conda install -c conda-forge pymol-open-source
```

**Note for different CUDA versions:** We currently don't yet provide package lists for different CUDA versions. If you use a different version, please use pip or conda to install the following packages:
```
torch (1.11)
dgl
pyg (aka torch-geometric)
e3nn
psutil
transformers
sentencepiece
biopython
matplotlib
```

## Regular prediction mode

You can compute structure predictions based on FASTA sequences using
```
python predict.py -i <FASTA> -o <OUTPUT_DIRECTORY>
```

The ProtT5 protein language model used to generate sequence embeddings will be downloaded on first use (~7 GB) and stored by default in the directory 'ProtT5-XL-U50'. You can change this directory with the `--t5_model` parameter.
By default, the script will produce PDB files and distance maps. You can disable outputs using the parameters `--no-pdb` and `--no-distance-maps` respectively.

## Mutation effect prediction

You can predict structures for all single amino-acid variants (SAVs) for the sequence(s) in a FASTA file using
```
python predict_sav.py -i <FASTA> -o <OUTPUT_DIRECTORY>
```

In addition to PDB files and distance maps, the SAV prediction script computes the structural difference between predictions for each mutant and the wild-type measured in lDDT (1.0 = most similar, 0.0 = least similar). These structure deltas are both rendered as an image (mutation_matrix.png) as well as provided as a text file for downstream consumption (mutation_log.txt).

## Mutation movie rendering

From previously computed SAV predictions (see above), you can render movies using
```
python render_mutation_movie.py <FASTA> <OUTPUT_DIRECTORY>
```

Alternatively, you can do both steps (prediction + movie rendering) at once using
```
./create_SAV_movie.sh <FASTA> <OUTPUT_DIRECTORY>
```

## Webserver mode

You can run a simple webserver for the visualization of predictions by starting
```
python webserver.py
```
and directing your browser at `http://localhost:24398/` or using the address of the machine the server is running on. You can change the default port number using the `-d` parameter when starting the webserver.

## Acknowledgements

We reused several modules of the [RoseTTAFold](https://github.com/RosettaCommons/RoseTTAFold) architecture. We use the SE(3)-Transformer implementation from [NVIDIA](https://github.com/NVIDIA/DeepLearningExamples/tree/master/DGLPyTorch/DrugDiscovery/SE3Transformer).

## Citing

For now, please cite this work as follows:
```
@software{Weissenow_EMBER3D_2022,
  author = {Weissenow, Konstantin and Heinzinger, Michael and Rost, Burkhard},
  doi = {10.5281/zenodo.6837687},
  month = {7},
  title = {{EMBER3D}},
  url = {https://github.com/kWeissenow/EMBER3D},
  year = {2022}
}
```
