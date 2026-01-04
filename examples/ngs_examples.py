"""
SSN Examples for NGS and Single-Cell Analysis

Demonstrates token-efficient notation for genomics workflows.
"""

import json
from ssn import SSN


def register_ngs_schemas(ssn: SSN):
    """Register NGS-specific schemas."""
    
    # RNA-seq
    ssn.register_schema(
        "rnaseq", "rna_sequencing",
        ["fastq_r1", "fastq_r2", "reference"],
        {"aligner": "star", "quantifier": "salmon"}
    )
    
    # Single-cell RNA-seq
    ssn.register_schema(
        "scrna", "single_cell_rnaseq",
        ["input_path", "reference"],
        {"tool": "cellranger", "chemistry": "auto"}
    )
    
    # ATAC-seq
    ssn.register_schema(
        "atac", "atac_sequencing",
        ["fastq_r1", "fastq_r2", "reference"],
        {"aligner": "bowtie2", "peak_caller": "macs2"}
    )
    
    # Variant calling
    ssn.register_schema(
        "variant", "variant_calling",
        ["bam", "reference"],
        {"caller": "gatk", "mode": "germline"}
    )
    
    # ChIP-seq
    ssn.register_schema(
        "chip", "chip_sequencing",
        ["treatment", "control", "reference"],
        {"peak_type": "narrow", "caller": "macs2"}
    )
    
    # Scanpy workflow
    ssn.register_schema(
        "scanpy", "scanpy_analysis",
        ["h5ad_file"],
        {"n_neighbors": 15, "n_pcs": 50}
    )
    
    return ssn


def example_rnaseq_pipeline():
    """Bulk RNA-seq analysis pipeline."""
    
    ssn = register_ngs_schemas(SSN())
    
    ssn_text = """
@rnaseq|sample1_R1.fastq.gz|sample1_R2.fastq.gz|GRCh38
>aligner:star
>quantifier:salmon
#paired_end
#stranded
.qc
  @run|fastqc
  @run|multiqc
  #trim_adapters
  >min_length:36
  >quality:20
.alignment
  >threads:16
  >ram:64G
  #two_pass
  >out_sam_type:BAM SortedByCoordinate
.quantification
  >library_type:ISR
  #gc_bias
  #seq_bias
  >bootstrap:100
.de_analysis
  >tool:deseq2
  >design:~condition
  >contrast:treated,control
  >fdr:0.05
  >lfc:1.0
"""
    
    json_data = {
        "pipeline": "rna_sequencing",
        "input": {
            "fastq_r1": "sample1_R1.fastq.gz",
            "fastq_r2": "sample1_R2.fastq.gz",
            "reference": "GRCh38"
        },
        "aligner": "star",
        "quantifier": "salmon",
        "paired_end": True,
        "stranded": True,
        "qc": {
            "tools": ["fastqc", "multiqc"],
            "trim_adapters": True,
            "min_length": 36,
            "quality": 20
        },
        "alignment": {
            "threads": 16,
            "ram": "64G",
            "two_pass": True,
            "out_sam_type": "BAM SortedByCoordinate"
        },
        "quantification": {
            "library_type": "ISR",
            "gc_bias": True,
            "seq_bias": True,
            "bootstrap": 100
        },
        "de_analysis": {
            "tool": "deseq2",
            "design": "~condition",
            "contrast": ["treated", "control"],
            "fdr": 0.05,
            "lfc": 1.0
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("=== RNA-seq Pipeline Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_single_cell_cellranger():
    """Single-cell RNA-seq with Cell Ranger."""
    
    ssn = register_ngs_schemas(SSN())
    
    ssn_text = """
@scrna|/data/fastqs|GRCh38
>sample_id:PBMC_10k
>chemistry:SC3Pv3
>expect_cells:10000
>localcores:16
>localmem:128
#include_introns
.aggr
  >samples:sample1,sample2,sample3
  >normalize:mapped
.reanalyze
  >num_pcs:50
  >max_clusters:15
  #no_secondary
"""
    
    json_data = {
        "pipeline": "single_cell_rnaseq",
        "tool": "cellranger",
        "input_path": "/data/fastqs",
        "reference": "GRCh38",
        "sample_id": "PBMC_10k",
        "chemistry": "SC3Pv3",
        "expect_cells": 10000,
        "localcores": 16,
        "localmem": 128,
        "include_introns": True,
        "aggr": {
            "samples": ["sample1", "sample2", "sample3"],
            "normalize": "mapped"
        },
        "reanalyze": {
            "num_pcs": 50,
            "max_clusters": 15,
            "no_secondary": True
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Single-Cell Cell Ranger Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_scanpy_workflow():
    """Scanpy single-cell analysis workflow."""
    
    ssn = register_ngs_schemas(SSN())
    
    ssn_text = """
@scanpy|pbmc_10k.h5ad
.preprocessing
  >min_genes:200
  >max_genes:5000
  >max_mt_pct:20
  >min_cells:3
  #filter_genes
  #normalize_total
  >target_sum:10000
  #log1p
  #highly_variable
  >n_top_genes:2000
  >flavor:seurat_v3
.dim_reduction
  #scale
  #pca
  >n_comps:50
  #neighbors
  >n_neighbors:15
  >metric:cosine
  #umap
  >min_dist:0.3
.clustering
  >method:leiden
  >resolution:0.5,0.8,1.0,1.2
  #rank_genes
  >n_genes:100
  >method:wilcoxon
.annotation
  >reference:celltypist
  >model:Immune_All_Low
  #majority_voting
.trajectory
  >method:paga
  >root:CD34+ HSC
  #diffmap
  >n_comps:15
  #dpt
"""
    
    json_data = {
        "pipeline": "scanpy_analysis",
        "input": "pbmc_10k.h5ad",
        "preprocessing": {
            "min_genes": 200,
            "max_genes": 5000,
            "max_mt_pct": 20,
            "min_cells": 3,
            "filter_genes": True,
            "normalize_total": True,
            "target_sum": 10000,
            "log1p": True,
            "highly_variable": True,
            "n_top_genes": 2000,
            "flavor": "seurat_v3"
        },
        "dim_reduction": {
            "scale": True,
            "pca": True,
            "n_comps": 50,
            "neighbors": True,
            "n_neighbors": 15,
            "metric": "cosine",
            "umap": True,
            "min_dist": 0.3
        },
        "clustering": {
            "method": "leiden",
            "resolution": [0.5, 0.8, 1.0, 1.2],
            "rank_genes": True,
            "n_genes": 100,
            "deg_method": "wilcoxon"
        },
        "annotation": {
            "reference": "celltypist",
            "model": "Immune_All_Low",
            "majority_voting": True
        },
        "trajectory": {
            "method": "paga",
            "root": "CD34+ HSC",
            "diffmap": True,
            "diffmap_n_comps": 15,
            "dpt": True
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Scanpy Workflow Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_seurat_workflow():
    """Seurat single-cell analysis workflow."""
    
    ssn = register_ngs_schemas(SSN())
    
    ssn_text = """
@seurat|filtered_feature_bc_matrix
>project:PBMC_Analysis
.qc
  >min_features:200
  >max_features:5000
  >max_mt:20
  #subset
.normalize
  >method:SCTransform
  >vars_to_regress:percent.mt,S.Score,G2M.Score
  #vst
  >variable_features:3000
.integrate
  >method:rpca
  >reference:1,2
  >k_anchor:5
  >dims:1:30
  #split_by:batch
.cluster
  >dims:1:30
  >resolution:0.4,0.6,0.8
  >algorithm:louvain
  #find_markers
  >test:wilcox
  >logfc:0.25
  >min_pct:0.1
.visualize
  #umap
  #tsne
  >perplexity:30
  #feature_plot
  >genes:CD3D,CD14,MS4A1,GNLY
"""
    
    json_data = {
        "pipeline": "seurat",
        "input": "filtered_feature_bc_matrix",
        "project": "PBMC_Analysis",
        "qc": {
            "min_features": 200,
            "max_features": 5000,
            "max_mt": 20,
            "subset": True
        },
        "normalize": {
            "method": "SCTransform",
            "vars_to_regress": ["percent.mt", "S.Score", "G2M.Score"],
            "vst": True,
            "variable_features": 3000
        },
        "integrate": {
            "method": "rpca",
            "reference": [1, 2],
            "k_anchor": 5,
            "dims": "1:30",
            "split_by": "batch"
        },
        "cluster": {
            "dims": "1:30",
            "resolution": [0.4, 0.6, 0.8],
            "algorithm": "louvain",
            "find_markers": True,
            "test": "wilcox",
            "logfc": 0.25,
            "min_pct": 0.1
        },
        "visualize": {
            "umap": True,
            "tsne": True,
            "perplexity": 30,
            "feature_plot": True,
            "genes": ["CD3D", "CD14", "MS4A1", "GNLY"]
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Seurat Workflow Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_multiome():
    """Multiome (RNA + ATAC) analysis."""
    
    ssn = register_ngs_schemas(SSN())
    
    ssn_text = """
@multiome|/data/multiome_fastqs|GRCh38
>sample:PBMC_multiome
>chemistry:ARC-v1
#include_introns
.rna
  @scanpy|~gex_h5
  >min_genes:200
  >n_top_genes:3000
  #normalize
  #pca
.atac
  @signac|~atac_fragments
  >min_peaks:1000
  >max_peaks:100000
  #lsi
  >n_components:50
  #call_peaks
  >method:macs2
.integration
  >method:WNN
  >rna_weight:0.5
  >atac_weight:0.5
  #joint_umap
  >n_neighbors:30
.linkage
  #gene_activity
  #peak_gene_links
  >distance:500000
  >correlation:0.3
"""
    
    json_data = {
        "pipeline": "multiome",
        "input": "/data/multiome_fastqs",
        "reference": "GRCh38",
        "sample": "PBMC_multiome",
        "chemistry": "ARC-v1",
        "include_introns": True,
        "rna": {
            "tool": "scanpy",
            "min_genes": 200,
            "n_top_genes": 3000,
            "normalize": True,
            "pca": True
        },
        "atac": {
            "tool": "signac",
            "min_peaks": 1000,
            "max_peaks": 100000,
            "lsi": True,
            "n_components": 50,
            "call_peaks": True,
            "peak_method": "macs2"
        },
        "integration": {
            "method": "WNN",
            "rna_weight": 0.5,
            "atac_weight": 0.5,
            "joint_umap": True,
            "n_neighbors": 30
        },
        "linkage": {
            "gene_activity": True,
            "peak_gene_links": True,
            "distance": 500000,
            "correlation": 0.3
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Multiome Analysis Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_spatial_transcriptomics():
    """Spatial transcriptomics analysis."""
    
    ssn = register_ngs_schemas(SSN())
    
    ssn_text = """
@spatial|visium_data|GRCh38
>sample:tumor_section_1
>slide:V10A20-016
>area:B1
#image_alignment
.preprocessing
  >min_counts:500
  >min_genes:250
  >max_mt:25
  #filter_spots
  #normalize
  >target_sum:10000
.clustering
  >n_neighbors:15
  >resolution:0.8
  #spatial_neighbors
  >coord_type:generic
  >n_rings:2
.deconvolution
  >method:cell2location
  >reference:sc_reference.h5ad
  >epochs:30000
  >lr:0.002
.spatial_analysis
  #autocorrelation
  >method:moran
  #co_occurrence
  >cluster_key:cell_type
  #niche
  >radius:100
  >min_cells:5
.visualization
  #spatial_scatter
  >color:leiden
  >img_alpha:0.5
  #spatial_genes
  >genes:ERBB2,ESR1,PGR,MKI67
"""
    
    json_data = {
        "pipeline": "spatial_transcriptomics",
        "platform": "visium",
        "input": "visium_data",
        "reference": "GRCh38",
        "sample": "tumor_section_1",
        "slide": "V10A20-016",
        "area": "B1",
        "image_alignment": True,
        "preprocessing": {
            "min_counts": 500,
            "min_genes": 250,
            "max_mt": 25,
            "filter_spots": True,
            "normalize": True,
            "target_sum": 10000
        },
        "clustering": {
            "n_neighbors": 15,
            "resolution": 0.8,
            "spatial_neighbors": True,
            "coord_type": "generic",
            "n_rings": 2
        },
        "deconvolution": {
            "method": "cell2location",
            "reference": "sc_reference.h5ad",
            "epochs": 30000,
            "learning_rate": 0.002
        },
        "spatial_analysis": {
            "autocorrelation": True,
            "method": "moran",
            "co_occurrence": True,
            "cluster_key": "cell_type",
            "niche": True,
            "radius": 100,
            "min_cells": 5
        },
        "visualization": {
            "spatial_scatter": True,
            "color": "leiden",
            "img_alpha": 0.5,
            "spatial_genes": True,
            "genes": ["ERBB2", "ESR1", "PGR", "MKI67"]
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Spatial Transcriptomics Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_variant_calling():
    """Variant calling pipeline."""
    
    ssn = register_ngs_schemas(SSN())
    
    ssn_text = """
@variant|aligned.bam|GRCh38
>caller:gatk
>mode:germline
.preprocessing
  @run|MarkDuplicates
  @run|BaseRecalibrator
  >known_sites:dbsnp,mills,1000g
  #spark
.calling
  >caller:HaplotypeCaller
  >ploidy:2
  >emit_ref_confidence:GVCF
  >intervals:exome_targets.bed
  #native_pair_hmm
.joint_genotyping
  >gvcfs:sample1.g.vcf,sample2.g.vcf,sample3.g.vcf
  #genomics_db
.filtering
  >mode:VQSR
  >truth:hapmap,omni,1000g,dbsnp
  >tranches:99.9,99.0,90.0
  #snp_model
  #indel_model
.annotation
  >tools:vep,snpeff
  >assembly:GRCh38
  #loftee
  #spliceai
  >database:clinvar,gnomad
"""
    
    json_data = {
        "pipeline": "variant_calling",
        "input": "aligned.bam",
        "reference": "GRCh38",
        "caller": "gatk",
        "mode": "germline",
        "preprocessing": {
            "steps": ["MarkDuplicates", "BaseRecalibrator"],
            "known_sites": ["dbsnp", "mills", "1000g"],
            "spark": True
        },
        "calling": {
            "caller": "HaplotypeCaller",
            "ploidy": 2,
            "emit_ref_confidence": "GVCF",
            "intervals": "exome_targets.bed",
            "native_pair_hmm": True
        },
        "joint_genotyping": {
            "gvcfs": ["sample1.g.vcf", "sample2.g.vcf", "sample3.g.vcf"],
            "genomics_db": True
        },
        "filtering": {
            "mode": "VQSR",
            "truth_resources": ["hapmap", "omni", "1000g", "dbsnp"],
            "tranches": [99.9, 99.0, 90.0],
            "snp_model": True,
            "indel_model": True
        },
        "annotation": {
            "tools": ["vep", "snpeff"],
            "assembly": "GRCh38",
            "loftee": True,
            "spliceai": True,
            "database": ["clinvar", "gnomad"]
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Variant Calling Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_atacseq():
    """ATAC-seq analysis pipeline."""
    
    ssn = register_ngs_schemas(SSN())
    
    ssn_text = """
@atac|sample_R1.fastq.gz|sample_R2.fastq.gz|GRCh38
>aligner:bowtie2
>peak_caller:macs2
#paired_end
.preprocessing
  @run|trim_galore
  >quality:20
  >length:20
  #fastqc
.alignment
  >threads:16
  >max_insert:2000
  #very_sensitive
  #no_mixed
  #no_discordant
.filtering
  #remove_duplicates
  #remove_chrM
  #remove_blacklist
  >mapq:30
  #shift_reads
  >shift:+4,-5
.peak_calling
  >format:BAMPE
  >gsize:hs
  >qvalue:0.05
  #broad
  >broad_cutoff:0.1
.analysis
  #frip
  #tss_enrichment
  #fragment_size
  >motif_analysis:homer
  >differential:diffbind
"""
    
    json_data = {
        "pipeline": "atac_sequencing",
        "input": {
            "fastq_r1": "sample_R1.fastq.gz",
            "fastq_r2": "sample_R2.fastq.gz",
            "reference": "GRCh38"
        },
        "aligner": "bowtie2",
        "peak_caller": "macs2",
        "paired_end": True,
        "preprocessing": {
            "trimmer": "trim_galore",
            "quality": 20,
            "length": 20,
            "fastqc": True
        },
        "alignment": {
            "threads": 16,
            "max_insert": 2000,
            "very_sensitive": True,
            "no_mixed": True,
            "no_discordant": True
        },
        "filtering": {
            "remove_duplicates": True,
            "remove_chrM": True,
            "remove_blacklist": True,
            "mapq": 30,
            "shift_reads": True,
            "shift": ["+4", "-5"]
        },
        "peak_calling": {
            "format": "BAMPE",
            "gsize": "hs",
            "qvalue": 0.05,
            "broad": True,
            "broad_cutoff": 0.1
        },
        "analysis": {
            "frip": True,
            "tss_enrichment": True,
            "fragment_size": True,
            "motif_analysis": "homer",
            "differential": "diffbind"
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== ATAC-seq Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_scrna_integration():
    """Multi-sample single-cell integration."""
    
    ssn = register_ngs_schemas(SSN())
    
    ssn_text = """
@integrate|scvi
>samples:ctrl_1,ctrl_2,treat_1,treat_2
>batch_key:sample
>labels_key:cell_type
.preprocessing
  #concatenate
  >join:outer
  #hvg
  >n_top_genes:4000
  >batch_key:sample
  >flavor:seurat_v3
.model
  >n_hidden:128
  >n_latent:30
  >n_layers:2
  >dropout:0.1
  >gene_likelihood:nb
.training
  >max_epochs:400
  >early_stopping:true
  >batch_size:256
  #plan_kwargs
  >lr:0.001
  >weight_decay:0.0001
.analysis
  #get_latent
  #differential
  >groupby:condition
  >method:change
  >batch_correction:true
  #de_genes
  >fdr:0.05
  >lfc:0.5
"""
    
    json_data = {
        "pipeline": "single_cell_integration",
        "method": "scvi",
        "samples": ["ctrl_1", "ctrl_2", "treat_1", "treat_2"],
        "batch_key": "sample",
        "labels_key": "cell_type",
        "preprocessing": {
            "concatenate": True,
            "join": "outer",
            "hvg": True,
            "n_top_genes": 4000,
            "batch_key": "sample",
            "flavor": "seurat_v3"
        },
        "model": {
            "n_hidden": 128,
            "n_latent": 30,
            "n_layers": 2,
            "dropout": 0.1,
            "gene_likelihood": "nb"
        },
        "training": {
            "max_epochs": 400,
            "early_stopping": True,
            "batch_size": 256,
            "plan_kwargs": True,
            "lr": 0.001,
            "weight_decay": 0.0001
        },
        "analysis": {
            "get_latent": True,
            "differential": True,
            "groupby": "condition",
            "method": "change",
            "batch_correction": True,
            "de_genes": True,
            "fdr": 0.05,
            "lfc": 0.5
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== scRNA Integration Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_long_read_sequencing():
    """Long-read sequencing (ONT/PacBio) analysis."""
    
    ssn = register_ngs_schemas(SSN())
    
    ssn_text = """
@longread|sample.fastq.gz|GRCh38
>platform:ont
>basecaller:dorado
>model:dna_r10.4.1_e8.2_400bps_sup
.alignment
  >aligner:minimap2
  >preset:map-ont
  >threads:32
  #secondary:no
  #MD_tag
.sv_calling
  >caller:sniffles2
  >min_support:3
  >min_sv_length:50
  #mosaic
  >tandem_repeats:trf.bed
.methylation
  #modkit
  >threshold:0.75
  >motif:CG
  >combine_strands
  >bedmethyl
.phasing
  >tool:whatshap
  >vcf:variants.vcf.gz
  #tag_reads
  #haplotag
.assembly
  >tool:flye
  >genome_size:3g
  >coverage:40
  #polish
  >rounds:2
"""
    
    json_data = {
        "pipeline": "long_read_sequencing",
        "input": "sample.fastq.gz",
        "reference": "GRCh38",
        "platform": "ont",
        "basecaller": "dorado",
        "model": "dna_r10.4.1_e8.2_400bps_sup",
        "alignment": {
            "aligner": "minimap2",
            "preset": "map-ont",
            "threads": 32,
            "secondary": "no",
            "MD_tag": True
        },
        "sv_calling": {
            "caller": "sniffles2",
            "min_support": 3,
            "min_sv_length": 50,
            "mosaic": True,
            "tandem_repeats": "trf.bed"
        },
        "methylation": {
            "modkit": True,
            "threshold": 0.75,
            "motif": "CG",
            "combine_strands": True,
            "bedmethyl": True
        },
        "phasing": {
            "tool": "whatshap",
            "vcf": "variants.vcf.gz",
            "tag_reads": True,
            "haplotag": True
        },
        "assembly": {
            "tool": "flye",
            "genome_size": "3g",
            "coverage": 40,
            "polish": True,
            "rounds": 2
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Long-Read Sequencing Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


def example_nextflow_config():
    """Nextflow pipeline configuration in SSN."""
    
    ssn = register_ngs_schemas(SSN())
    
    ssn_text = """
@nextflow|nf-core/rnaseq
>revision:3.14.0
>profile:docker,test
.params
  >input:samplesheet.csv
  >outdir:results
  >genome:GRCh38
  >aligner:star_salmon
  #skip_qc:false
  #skip_trimming:false
  >min_mapped_reads:5
.resources
  >max_cpus:16
  >max_memory:128.GB
  >max_time:240.h
.executor
  >name:slurm
  >queue:normal
  >submit_rate_limit:10/1min
  #retry_on_fail
  >max_retries:3
.singularity
  #enabled
  >cache_dir:/scratch/singularity
  #auto_mounts
"""
    
    json_data = {
        "pipeline": "nf-core/rnaseq",
        "revision": "3.14.0",
        "profile": ["docker", "test"],
        "params": {
            "input": "samplesheet.csv",
            "outdir": "results",
            "genome": "GRCh38",
            "aligner": "star_salmon",
            "skip_qc": False,
            "skip_trimming": False,
            "min_mapped_reads": 5
        },
        "resources": {
            "max_cpus": 16,
            "max_memory": "128.GB",
            "max_time": "240.h"
        },
        "executor": {
            "name": "slurm",
            "queue": "normal",
            "submit_rate_limit": "10/1min",
            "retry_on_fail": True,
            "max_retries": 3
        },
        "singularity": {
            "enabled": True,
            "cache_dir": "/scratch/singularity",
            "auto_mounts": True
        }
    }
    
    json_text = json.dumps(json_data)
    
    print("\n=== Nextflow Config Example ===")
    print(f"\nSSN ({len(ssn_text)} chars):")
    print(ssn_text)
    print(f"\nJSON ({len(json_text)} chars):")
    print(json_text)
    
    stats = ssn.token_stats(ssn_text, json_text)
    print(f"\nToken reduction: {stats['reduction_percent']}%")


if __name__ == "__main__":
    example_rnaseq_pipeline()
    example_single_cell_cellranger()
    example_scanpy_workflow()
    example_seurat_workflow()
    example_multiome()
    example_spatial_transcriptomics()
    example_variant_calling()
    example_atacseq()
    example_scrna_integration()
    example_long_read_sequencing()
    example_nextflow_config()
