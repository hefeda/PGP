# PPIHP
Protein Prediction for Interpretation of Hallucinated Proteins

This script allows ultra-fast (around 30 minutes per proteome) prediction of various protein properties using the protein language model [ProtT5](https://github.com/agemagician/ProtTrans).
For an overview of currently supported predictors, check the [Outputs](https://github.com/hefeda/PGP#outputs) section.

Additionally, this script allows to generate completely novel protein sequences using [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2) which can be further filtered using the aforementioned predictions.

This tool which ties together many existing predictors was first introduced in [From sequence to function through structure: Deep learning for protein design](https://www.sciencedirect.com/science/article/pii/S2001037022005086).

## Installation

Create a new virtual environment, e.g. using conda:
```
conda create -n PPIHP python=3.8
conda activate PPIHP
```


**Without 3D structure prediction**:

If you do not need 3D structure predictions (avoids many dependencies)
```
pip install -r requirements_minimal.txt
```


**With 3D structure prediction:**  
If you use CUDA 11, you can use the provided requirements.txt to install dependencies for all predictors:
```
pip install -r requirements.txt
```

If you use a different version, please use pip or conda to install the following packages:
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


# Usage
If you pass an input FASTA, predictions for the given sequences will be generated and written to the directory defined by the output parameter (this will download around 3GB of model weights in total):
```sh
python prott5_batch_predictor.py --input example_output/pp_examples.fasta --output example_output
```

If you only pass an output directory without any input-FASTA, the script will generate new random proteins using [ProtGPT2](https://huggingface.co/nferruz/ProtGPT2) and generate predictions for those hallucinated proteins (this will download additionally 2.5GB model weights of ProtGPT2):
```sh
python prott5_batch_predictor.py --output halluzination_analysis --n_gen 50
```
The parameter `n_gen` allows you to control the number of sequences to generate.


If you only need a subset of predictors, you can adjust which predictors to run using the --fmt parameter:
```sh
python prott5_batch_predictor.py --output halluzination_analysis --n_gen 50 --fmt ss,cons,dis,mem,bind,go,subcell,tucker,emb,ember3D
```
This allows, for example, to (de-)activate 3D structure prediction. See --help for more information on the output format. By default, all predictors except 3D structure prediction and per-proteins embeddings are written (those quickly generate large amounts of data when applied to millions of proteins and should only be used with caution; default=`--fmt ss,cons,dis,mem,bind,go,subcell,tucker`.

## Reproducibility
The datasets used for the analysis in the manuscript are available at: http://data.bioembeddings.com/public/design/.
Place them into a folder called `private` inside this repo to run the Jupyter Notebooks.
Predictions were generated on a server with an Intel Xeon Gold 6248 CPU, a Quadro RTX 8000 (48GB vRAM) GPU and 400GB RAM DDR4 ECC (OS=Ubuntu).   

# Outputs
The current scripts generates a wealth of information for each input protein sequences.
Every predictor generates one output file.
In general, all files are written such that each line holds information on a single protein and sorting between files is identical.
The `ids.txt` file explained below allows to backtrace which prediction refers to which protein.
The 3D structure prediction is an exception to this schema as it writes one PDB-file per input protein.
The following files are currently generated:

**General:**

- `ids.txt`: each line holds the ID parsed from your input fasta (in case of halucinated proteins, this contains perplexity)
- `seqs.txt`: each line holds an amino acid sequence (either copied from input fasta or halucinated sequence)
- `protein_embeddings.npy`: numpy array holding per-protein embeddings generated via ProtT5-XL-U50 (or short [ProtT5](https://github.com/agemagician/ProtTrans)). Can be read in via `numpy.load("protein_embeddings.npy")`. Row sorting is identical to ordering of IDs in `ids.txt`

**3D Structure prediction:**
- `pdbs`: this directory holds one PDB file per input protein. Predictions are generated using [EMBER3D](https://github.com/kWeissenow/EMBER3D)



**Predictions available for each residue in a protein:**

- `conservation_pred.txt`: each line holds sequence conservation in [0,8] as defined by [ConSeq](https://academic.oup.com/bioinformatics/article/20/8/1322/209847?login=true). 0 refers to highly variable, 8 to highly conserved residues. Published in [VESPA](https://github.com/Rostlab/VESPA)
- `dssp3_pred.txt`: each line holds 3-state secondary structure prediction. H=Helix, E=Sheet, L=Other. Predictor published in [ProtTrans](https://github.com/agemagician/ProtTrans)
- `dssp8_pred.txt`: each line holds 8-state secondary structure prediction as defined by [DSSP](https://swift.cmbi.umcn.nl/gv/dssp/). Predictor published in [ProtTrans](https://github.com/agemagician/ProtTrans)
- `membrane_tmbed.txt`: each line holds transmembrane predictions generated from [TMbed](https://github.com/BernhoferM/TMbed). Output format is described under [`--out-format=1`](https://github.com/BernhoferM/TMbed#prediction-output)
- `seth_disorder_pred.csv`: each line holds disorder prediction in the form of [CheZOD scores](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4749933/) generated via [SETH](https://github.com/DagmarIlz/SETH). Scores are in [-17,17] with values >8 indicating order and <8 indicating disorder.
- `binding_bindEmbed_metal_pred.txt`: each line holds metal-binding ('M') or not ('-') predicted via [BindEmbed21DL](https://doi.org/10.1038/s41598-021-03431-4).
- `binding_bindEmbed_nucleic_pred.txt`: each line holds nucleic-acid-binding ('N') or not ('-') predicted via [BindEmbed21DL](https://doi.org/10.1038/s41598-021-03431-4).
- `binding_bindEmbed_small_pred.txt`: each line holds small-molecule-binding ('S') or not ('-') predicted via [BindEmbed21DL](https://doi.org/10.1038/s41598-021-03431-4).

**Predictions available for each protein:**
- `goPredSim_GO_bpo_pred.csv`: embedding-based annotation transfer (EAT) predicts GO-BPO via [goPredSim](https://doi.org/10.1038/s41598-020-80786-0). First element holds ID of closest SwissProt protein hit that was used for annotation transfer. Second field holds all available GO-BPO annotations for this lookup protein. (N/A if no prediction possible). Third field gives Euclidean distance between query and lookup (the closer/smaller the better/more_reliable). A reasonable threshold for annotation transfer is a Euclidean Distance <=0.5 .
- `goPredSim_GO_mfo_pred.csv`: same as GO-BPO but for MFO [goPredSim](https://doi.org/10.1038/s41598-020-80786-0) (N/A if no prediction possible)
- `goPredSim_GO_cco_pred.csv`: same as GO-BPO but for CCO [goPredSim](https://doi.org/10.1038/s41598-020-80786-0) (N/A if no prediction possible)
- `prottucker_CATH_pred.csv`: EAT predicts structural classes as defined by CATH via [ProtTucker](https://doi.org/10.1093/nargab/lqac043). First element holds CATH-ID of closest hit; second field holds CATH annotation of this hit. (N/A if no prediction possible). Third field gives Euclidean distance between query and lookup (the closer/smaller the better/more_reliable). A reasonable threshold for annotation transfer is a Euclidean Distance <=1.1 .
- `la_mem_pred.txt`: each line holds binary classification into membrane-bound vs water-soluble [LightAttention](https://doi.org/10.1093/bioadv/vbab035)
- `la_subcell_pred.txt`: each line holds predicted subcellular localization as defined by [LightAttention](https://doi.org/10.1093/bioadv/vbab035)


# Citations
- ProtT5 (Generates input embeddings for all predictors. Also, correct ref. for secondary structure prediction):
```Bibtex
@article{9477085,
author={Elnaggar, Ahmed and Heinzinger, Michael and Dallago, Christian and Rehawi, Ghalia and Yu, Wang and Jones, Llion and Gibbs, Tom and Feher, Tamas and Angerer, Christoph and Steinegger, Martin and Bhowmik, Debsindhu and Rost, Burkhard},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
title={ProtTrans: Towards Cracking the Language of Lifes Code Through Self-Supervised Deep Learning and High Performance Computing},
year={2021},
volume={},
number={},
pages={1-1},
doi={10.1109/TPAMI.2021.3095381}
}
```
- ProtGPT2 (sequence generation/halucination):
```Bibtex
@article{Ferruz2022.03.09.483666,
	author = {Ferruz, Noelia and Schmidt, Steffen and H{\"o}cker, Birte},
	title = {A deep unsupervised language model for protein design},
	elocation-id = {2022.03.09.483666},
	year = {2022},
	doi = {10.1101/2022.03.09.483666},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/03/12/2022.03.09.483666},
	eprint = {https://www.biorxiv.org/content/early/2022/03/12/2022.03.09.483666.full.pdf},
	journal = {bioRxiv}
}
```

- TMbed (transmembrane prediction):
```Bibtex
@article {Bernhofer2022.06.12.495804,
	author = {Bernhofer, Michael and Rost, Burkhard},
	title = {TMbed {\textendash} Transmembrane proteins predicted through Language Model embeddings},
	elocation-id = {2022.06.12.495804},
	year = {2022},
	doi = {10.1101/2022.06.12.495804},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/06/15/2022.06.12.495804},
	eprint = {https://www.biorxiv.org/content/early/2022/06/15/2022.06.12.495804.full.pdf},
	journal = {bioRxiv}
}
```

- VESPA (conservation prediction):
```Bibtex
@article{Marquet2021,
  doi = {10.1007/s00439-021-02411-y},
  url = {https://doi.org/10.1007/s00439-021-02411-y},
  year = {2021},
  month = dec,
  publisher = {Springer Science and Business Media {LLC}},
  author = {C{\'{e}}line Marquet and Michael Heinzinger and Tobias Olenyi and Christian Dallago and Kyra Erckert and Michael Bernhofer and Dmitrii Nechaev and Burkhard Rost},
  title = {Embeddings from protein language models predict conservation and variant effects},
  journal = {Human Genetics}
}
```

- SETH (disorder prediction):
```Bibtex
@article {Ilzhoefer2022.06.23.497276,
	author = {Ilzhoefer, Dagmar and Heinzinger, Michael and Rost, Burkhard},
	title = {SETH predicts nuances of residue disorder from protein embeddings},
	elocation-id = {2022.06.23.497276},
	year = {2022},
	doi = {10.1101/2022.06.23.497276},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2022/06/26/2022.06.23.497276},
	eprint = {https://www.biorxiv.org/content/early/2022/06/26/2022.06.23.497276.full.pdf},
	journal = {bioRxiv}
}
```

- goPredSim (GO prediction):
```Bibtex
@article{littmann2021embeddings,
  title={Embeddings from deep learning transfer GO annotations beyond homology},
  author={Littmann, Maria and Heinzinger, Michael and Dallago, Christian and Olenyi, Tobias and Rost, Burkhard},
  journal={Scientific reports},
  volume={11},
  number={1},
  pages={1--14},
  year={2021},
  publisher={Nature Publishing Group}
}
```

- ProtTucker (CATH prediction):
```Bibtex
@article{10.1093/nargab/lqac043,
    author = {Heinzinger, Michael and Littmann, Maria and Sillitoe, Ian and Bordin, Nicola and Orengo, Christine and Rost, Burkhard},
    title = "{Contrastive learning on protein embeddings enlightens midnight zone}",
    journal = {NAR Genomics and Bioinformatics},
    volume = {4},
    number = {2},
    year = {2022},
    month = {06},
    issn = {2631-9268},
    doi = {10.1093/nargab/lqac043},
    url = {https://doi.org/10.1093/nargab/lqac043},
    note = {lqac043},
    eprint = {https://academic.oup.com/nargab/article-pdf/4/2/lqac043/44245898/lqac043.pdf},
}
```

- LightAttention (Subcellular localization prediction & membrane-vs-soluble):
```Bibtex
@article{10.1093/bioadv/vbab035,
    author = {Stärk, Hannes and Dallago, Christian and Heinzinger, Michael and Rost, Burkhard},
    title = "{Light attention predicts protein location from the language of life}",
    journal = {Bioinformatics Advances},
    volume = {1},
    number = {1},
    year = {2021},
    month = {11},
    issn = {2635-0041},
    doi = {10.1093/bioadv/vbab035},
    url = {https://doi.org/10.1093/bioadv/vbab035},
    note = {vbab035},
    eprint = {https://academic.oup.com/bioinformaticsadvances/article-pdf/1/1/vbab035/41640353/vbab035.pdf},
}

```


- BindEmbed21DL (binding-residue prediction; metal-, small-mol., nucleic):
```Bibtex
@article{littmann2021protein,
  title={Protein embeddings and deep learning predict binding residues for various ligand classes},
  author={Littmann, Maria and Heinzinger, Michael and Dallago, Christian and Weissenow, Konstantin and Rost, Burkhard},
  journal={Scientific Reports},
  volume={11},
  number={1},
  pages={1--15},
  year={2021},
  publisher={Nature Publishing Group}
}
```

- EMBER3D (3D structure prediction):
```Bibtex
@software{Weissenow_EMBER3D_2022,
  author = {Weissenow, Konstantin and Heinzinger, Michael and Rost, Burkhard},
  doi = {10.5281/zenodo.6837687},
  month = {7},
  title = {{EMBER3D}},
  url = {https://github.com/kWeissenow/EMBER3D},
  year = {2022}
}
```

