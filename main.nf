nextflow.enable.dsl=2

/*
 * Glycosylation site prediction pipeline
 * ---------------------------------------
 * train()
 * 1. PREPROCESS : parse raw TSV, derive per-residue binary glycosylation labels
 * 2. SPLIT_TEST : split data into train+val and test sets
 * 3. EMBED      : generate ESMC-300M per-residue embeddings for each sequence
 * 4. TRAIN      : split embeddings into train/val, train residue classifier
 * predict()
 * 1. PREPROCESS : parse raw TSV, derive per-residue binary glycosylation labels
 * 2. EMBED      : generate ESMC-300M per-residue embeddings for each sequence
 * 3. PREDICT    : make predictions
 */

// General
params.entry        = null              // 'train' or 'predict'
params.input_tsv   = null               // path to raw TSV (Sequence + Glycosylation columns)
params.input_json  = null               // path to preprocessed JSON (from PREPROCESS)
params.model_ckpt = null                // path to a trained classifier.pt (from train.py), required for -entry predict
params.outdir      = "results"          // output directory

// Preprocessing
params.drop_non_glycosylated = true     // whether to drop sequences without glycosylation annotations
params.length_filter = 500              // filter out sequences longer than this length
params.split_test = true                // whether to split off a test set (10% of data) for evaluation
params.test_size    = 0.05              // proportion of data to split off as test set (only used if split_test is true)

// Embedding
params.batch_size_embed = 8             // batch size for embedding generation

// Training
params.val_size     = 0.15
params.random_state  = 42
params.num_epochs    = 10
params.train_batch_size = 4096          // batch size for classifier training
params.lr            = 0.0001
params.hidden_size   = 960              // ESMC-300M hidden dim
params.dropout       = 0.1
params.optimise_metric = 'f1'           // threshold optimisation metric

// Prediction
params.threshold_json = null            // path to threshold.json from a train() run (optional)
params.threshold      = 0.5             // fallback decision threshold if threshold_json isn't given


workflow {
    if (params.entry == 'predict') {
        predict()
    } else if (params.entry == 'train') {
        train()
    } else {
        error "Please provide --entry <train|predict>"
    }
}

workflow train {

    if ((!params.input_tsv)&&(!params.input_json)) {
        error "Please provide --input_tsv <path to raw TSV file> or --input_json <path ot preprocessed JSON>"
    }

    input_ch = Channel.fromPath(params.input_tsv, checkIfExists: true)

    if (params.input_tsv) {
        PREPROCESS(input_ch)
    }
    
    if (params.split_test) {
        SPLIT_TEST(PREPROCESS.out.processed)
        EMBED(SPLIT_TEST.out.train_val)
    } else {
        EMBED(PREPROCESS.out.processed)
    }

    TRAIN(EMBED.out.embedding)
}

workflow predict {
 
    if ((!params.input_tsv)&&(!params.input_json)) {
        error "Please provide --input_tsv <path to raw TSV file> or --input_json <path ot preprocessed JSON>"
    }
    if (!params.model_ckpt) {
        error "Please provide --model_ckpt <path to trained classifier.pt>"
    }
    model_ch = Channel.fromPath(params.model_ckpt, checkIfExists: true)

    def threshold_value = params.threshold
    if (params.threshold_json) {
        def threshold_file = file(params.threshold_json)
        if (!threshold_file.exists()) {
            error "threshold_json file not found: ${params.threshold_json}"
        }
        threshold_value = new groovy.json.JsonSlurper().parse(threshold_file)['threshold']
        log.info "Using threshold ${threshold_value} loaded from ${params.threshold_json}"
    } else {
        log.info "No --threshold_json provided; using --threshold default (${threshold_value})"
    }
    threshold_ch = Channel.value(threshold_value)
    
    if (params.input_json) {
        processed_ch = Channel.fromPath(params.input_json, checkIfExists: true)
    } else {
        input_ch = Channel.fromPath(params.input_tsv, checkIfExists: true)
        PREPROCESS(input_ch)
        processed_ch = PREPROCESS.out.processed
    }
    EMBED(processed_ch)
    PREDICT(EMBED.out.embedding, model_ch, threshold_ch)
}

process PREPROCESS {
    tag "preprocess"
    publishDir "${params.outdir}/preprocess", mode: 'copy'

    input:
    path raw_tsv

    output:
    path "processed.json", emit: processed
    path "preprocess.log", emit: log

    script:
    """
    preprocess.py \\
        --input ${raw_tsv} \\
        --output processed.json \\
        ${params.drop_non_glycosylated ? '--drop_non_glycosylated' : ''} \\
        --length_filter ${params.length_filter} 2>&1 | tee preprocess.log
    """
}

process SPLIT_TEST {
    tag "split-test"
    publishDir "${params.outdir}/split-test", mode: 'copy'

    input:
    path processed_json

    output:
    path "processed-train-val.json", emit: train_val
    path "processed-test.json", emit: test
    path "split-test.log", emit: log

    script:
    """
    split-test.py \\
        --input ${processed_json} \\
        --train_val_out processed-train-val.json \\
        --test_out processed-test.json \\
        --test_size ${params.test_size} \\
        --random_state ${params.random_state} 2>&1 | tee split-test.log
    """
}

process EMBED {
    tag "embed"
    publishDir "${params.outdir}/embeddings", mode: 'copy'
    accelerator 1
    debug true

    input:
    path processed_json

    output:
    path "embedding-data.pt", emit: embedding
    path "embed.log", emit: log

    script:
    """
    get-embeddings.py \\
        --input ${processed_json} \\
        --output embedding-data.pt \\
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
    path "classifier.pt", emit: model
    path "threshold.json", emit: threshold
    path "history.json"
    path "train.log"

    script:
    """
    train.py \\
        --input ${embedding_data} \\
        --model_out classifier.pt \\
        --history_out history.json \\
        --threshold_out threshold.json \\
        --val_size ${params.val_size} \\
        --test_size ${params.test_size} \\
        --random_state ${params.random_state} \\
        --num_epochs ${params.num_epochs} \\
        --batch_size ${params.train_batch_size} \\
        --lr ${params.lr} \\
        --hidden_size ${params.hidden_size} \\
        --optimise_metric ${params.optimise_metric} 2>&1 | tee train.log
    """
}

process PREDICT {
    tag "predict"
    publishDir "${params.outdir}/predictions", mode: 'copy'
    debug true
    accelerator 1
 
    input:
    path embedding_data
    path model_ckpt
    val decision_threshold
 
    output:
    path "predictions.json", emit: predictions
    path "test-metrics.json", emit: metrics, optional: true
 
    script:
    """
    predict.py \\
        --embedding_input ${embedding_data} \\
        --model_ckpt ${model_ckpt} \\
        --hidden_size ${params.hidden_size} \\
        --threshold ${decision_threshold} \\
        --predictions_out predictions.json \\
        --metrics_out test-metrics.json 2>&1 | tee predict.log
    """
}