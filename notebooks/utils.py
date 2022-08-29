from functools import reduce
from pathlib import Path

from pandas import read_csv, DataFrame, merge

from itertools import groupby

def load_data(output_directory: Path) -> DataFrame:
    indexes = read_csv(str(output_directory / 'ids.txt'), names=['header'], sep="\t")
    sequences = read_csv(str(output_directory / 'seqs.txt'), names=['sequence'], sep="\t")

    sequences['length'] = sequences.sequence.str.len()
    try:
        sequences['ppl'] = indexes.header.str.split(",").map(lambda e: float(e[2].replace("ppl=","")))
    except (ValueError, IndexError):
        print("No perplexity found in header. Skipping this field.")
        pass
    seq_lenghts = sequences['length'].values
    
    disorder = read_csv(str(output_directory / 'seth_disorder_pred.csv'), names=['disorder'], sep="\t")
    
    disorder.disorder = disorder.disorder.str.split(r",\s*").map(lambda e: [float(n) for n in e])
    disorder['disorder_categorical'] = disorder.disorder.map(lambda e: "".join(["D" if n<8 else "-" for n in e]))
    disorder['disorder_average'] = disorder.disorder.map(lambda e: sum(e)/len(e))
    disorder['disorder_count'] = disorder.disorder_categorical.str.count("D")
    disorder['disorder_percent'] = disorder['disorder_count']/seq_lenghts
    # Count disorder stretches
    disorder['disorder_stretches'] = disorder.disorder_categorical.map(
        lambda e: [sum(1 for _ in group) for label, group in groupby(e) if label=="D"])
    
    metal = read_csv(str(output_directory / 'binding_bindEmbed_metal_pred.txt'), names=['metal'], sep="\t")
    
    metal['metal_count'] = metal.metal.str.count("M")
    metal['metal_percent'] = metal['metal_count']/seq_lenghts
    
    small = read_csv(str(output_directory / 'binding_bindEmbed_small_pred.txt'), names=['small'], sep="\t")
    
    small['small_count'] = small.small.str.count("S")
    small['small_percent'] = small['small_count']/seq_lenghts
    
    nucleic = read_csv(str(output_directory / 'binding_bindEmbed_nucleic_pred.txt'), names=['nucleic'], sep="\t")
    
    nucleic['nucleic_count'] = nucleic.nucleic.str.count("N")
    nucleic['nucleic_percent'] = nucleic['nucleic_count']/seq_lenghts
    
    conservation = read_csv(str(output_directory / 'conservation_pred.txt'), names=['conservation'], sep="\t")
    
    conservation['conservation_categorical'] = conservation.conservation.str.replace(r"[0-2]", 'L', regex=True).str.replace(r"[3-5]", 'M', regex=True).str.replace(r"[6-9]", 'H', regex=True)

    conservation['conservation_high_count'] = conservation.conservation_categorical.str.count('H')
    conservation['conservation_high_percent'] = conservation['conservation_high_count']/seq_lenghts
    conservation['conservation_low_count'] = conservation.conservation_categorical.str.count('L')
    conservation['conservation_low_percent'] = conservation['conservation_low_count']/seq_lenghts

    dssp3 = read_csv(str(output_directory / 'dssp3_pred.txt'), names=['dssp3'], sep="\t")
    
    dssp3['helix_count'] = dssp3.dssp3.str.count("H")
    dssp3['helix_percent'] = dssp3['helix_count']/seq_lenghts
    
    dssp3['strand_count'] = dssp3.dssp3.str.count("E")
    dssp3['strand_percent'] = dssp3['strand_count']/seq_lenghts
    dssp3['strand_stretch_count'] = dssp3.dssp3.map(
        lambda e: [sum(1 for _ in group) for label, group in groupby(e) if label=="E"])
    
    
    dssp3['helix_four_count'] = dssp3.dssp3.str.count(r"H{4}")
    dssp3['helix_four_percent'] = dssp3['helix_four_count']/(seq_lenghts/4)
    
    dssp3['helix_stretch_count'] = dssp3.dssp3.map(
        lambda e: [sum(1 for _ in group) for label, group in groupby(e) if label=="H"])
    
    bpo = read_csv(str(output_directory / 'goPredSim_GO_bpo_pred.csv'), names=['BPO_reference', 'BPO_terms', "BPO_distance"], sep="\t")

    cco = read_csv(str(output_directory / 'goPredSim_GO_cco_pred.csv'), names=['CCO_reference', 'CCO_terms', "CCO_distance"], sep="\t")
    
    mfo = read_csv(str(output_directory / 'goPredSim_GO_mfo_pred.csv'), names=['MFO_reference', 'MFO_terms', "MFO_distance"], sep="\t")
    
    subcell = read_csv(str(output_directory / 'la_subcell_pred.txt'), names=['subcellular_location'], sep="\t")

    cath = read_csv(str(output_directory / 'prottucker_CATH_pred.csv'), names=['CATH_reference', 'CATH_superfamily', 'CATH_distance'], sep="\t")
    
    transmembrane = read_csv(str(output_directory / 'membrane_tmbed.txt'), names=['transmembrane'], sep="\t")
    
    transmembrane['signal_residue_count'] = transmembrane.transmembrane.str.count("S")
    # A signal peptide should probably be >=1 residues (probably rather min 5-11 but unsure about lower bound of SP-length so let's stick to 1 for simplicity)
    transmembrane['signal_protein'] = transmembrane.transmembrane.map(lambda e: "With SP" if e.count("S") > 1 else "Without SP" )
    transmembrane['signal_residue_percent'] = transmembrane['signal_residue_count']/seq_lenghts

    transmembrane['transmembrane_helix_count'] = transmembrane.transmembrane.str.count(r"[h|H]")
    transmembrane['transmembrane_helix_percent'] = transmembrane['transmembrane_helix_count']/seq_lenghts
    # Count disorder stretches
    transmembrane['transmembrane_helix_stretches'] = transmembrane.transmembrane.map(
        lambda e: [sum(1 for _ in group) for label, group in groupby(e) if label=="H" or label=="h"])


    transmembrane['transmembrane_strand_count'] = transmembrane.transmembrane.str.count(r"[b|B]")
    transmembrane['transmembrane_strand_percent'] = transmembrane['transmembrane_strand_count']/seq_lenghts
    # Count disorder stretches
    transmembrane['transmembrane_strand_stretches'] = transmembrane.transmembrane.map(
        lambda e: [sum(1 for _ in group) for label, group in groupby(e) if label=="B" or label=="b"])


    return reduce(lambda  left, right: merge(left, right, left_index=True, right_index=True, how='outer'), [indexes, sequences, disorder, metal, small, nucleic, conservation, dssp3, bpo, cco, mfo, subcell, cath, transmembrane])
