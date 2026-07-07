nextflow.enable.dsl=2

/*
 * Glycosylation site prediction pipeline
 * ---------------------------------------
 * 1. PREPROCESS : parse raw TSV, derive per-residue binary glycosylation labels
 * 2. EMBED      : generate ESMC-300M per-residue embeddings for each sequence
 * 3. TRAIN      : split embeddings into train/val, train residue classifier
 */

params.input_tsv   = null          // path to raw TSV (Sequence + Glycosylation columns)
params.outdir      = "results"     // output directory
params.batch_size_embed = 8        // batch size for embedding generation
params.val_size     = 0.2
params.random_state  = 42
params.num_epochs    = 5
params.train_batch_size = 8
params.lr            = 0.0001
params.hidden_size   = 960         // ESMC-300M hidden dim

workflow {

    if (!params.input_tsv) {
        error "Please provide --input_tsv <path to raw TSV file>"
    }

    input_ch = Channel.fromPath(params.input_tsv, checkIfExists: true)

    PREPROCESS(input_ch)
    EMBED(PREPROCESS.out.processed)
    TRAIN(EMBED.out.embedding)
}

process PREPROCESS {
    tag "preprocess"
    publishDir "${params.outdir}/preprocess", mode: 'copy'

    input:
    path raw_tsv

    output:
    path "processed.pkl", emit: processed
    path "preprocess.log", emit: log

    script:
    """
    preprocess.py --input ${raw_tsv} --output processed.pkl 2>&1 | tee preprocess.log
    """
}

process EMBED {
    tag "embed"
    publishDir "${params.outdir}/embeddings", mode: 'copy'
    accelerator 1

    input:
    path processed_pkl

    output:
    path "embedding_data.pt", emit: embedding
    path "embed.log", emit: log

    script:
    """
    get_embeddings.py \\
        --input ${processed_pkl} \\
        --output embedding_data.pt \\
        --batch_size ${params.batch_size_embed} 2>&1 | tee embed.log
    """
}

process TRAIN {
    tag "train"
    publishDir "${params.outdir}/model", mode: 'copy'
    accelerator 1

    input:
    path embedding_data

    output:
    path "classifier.pt"
    path "history.json"
    path "train.log"

    script:
    """
    train.py \\
        --input ${embedding_data} \\
        --model_out classifier.pt \\
        --history_out history.json \\
        --val_size ${params.val_size} \\
        --random_state ${params.random_state} \\
        --num_epochs ${params.num_epochs} \\
        --batch_size ${params.train_batch_size} \\
        --lr ${params.lr} \\
        --hidden_size ${params.hidden_size} 2>&1 | tee train.log
    """
}