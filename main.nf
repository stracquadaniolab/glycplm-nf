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
params.entry                    = null                          // 'train' or 'predict'
params.input_tsv                = null                          // path to raw TSV (Sequence + Glycosylation columns)
params.outdir                   = "/localdisk/storage/projects/glycplm-nf/results/results-0.1.2-1" // output directory
params.cache_dir                = "/localdisk/storage/projects/glycplm-nf/cache" // dir for preprocess/split-test/embed outputs
params.input_json               = null                          // skip PREPROCESS
params.embedding_pt             = null                          // skip PREPROCESS + EMBED entirely
params.model_ckpt               = "${params.outdir}/train/model/classifier.pt" // path to a trained classifier.pt (from train.py), required for -entry predict

// Preprocessing
params.drop_non_glycosylated    = true                          // whether to drop sequences without glycosylation annotations
params.length_filter            = 500                           // filter out sequences longer than this length
params.split_test               = true                          // whether to split off a test set (10% of data) for evaluation
params.test_size                = 0.05                          // proportion of data to split off as test set (only used if split_test is true)

// Embedding
params.batch_size_embed         = 8                             // batch size for embedding generation

// Training
params.val_size                 = 0.15
params.random_state             = 42
params.num_epochs               = 10
params.train_batch_size         = 4096                          // batch size for classifier training
params.lr                       = 0.0001
params.hidden_size              = 960                           // ESMC-300M hidden dim
params.dropout                  = 0.1
params.optimise_metric          = 'f1'                          // threshold optimisation metric
params.classifier               = 'LL'                          // whether to use single linear layer or MLP
params.mlp_hidden_size          = 256

// Prediction
params.threshold_json           = "${params.outdir}/train/model/threshold.json" // path to threshold.json from a train() run (optional)
params.threshold                = 0.5                           // fallback decision threshold if threshold_json isn't given


workflow {

    file(params.outdir).mkdirs()
    file("${params.outdir}/params.log").text = params
        .findAll { k, v -> k != 'class' }
        .collect { k, v -> "${k} = ${v}" }
        .sort()
        .join("\n")

    if (params.entry == 'predict') {
        predict()
    } else if (params.entry == 'train') {
        train()
    } else {
        error "Please provide --entry <train|predict>"
    }
}

workflow train {

    if (params.embedding_pt) {
        embedding_ch = Channel.fromPath(params.embedding_pt, checkIfExists: true)
    } else {
        if ((!params.input_tsv)&&(!params.input_json)) {
        error "Please provide --input_tsv <path to raw TSV file> or --input_json <path ot preprocessed JSON>"
        }

        input_ch = Channel.fromPath(params.input_tsv, checkIfExists: true)

        if (params.input_tsv) {
            PREPROCESS(input_ch)
        }

        if (params.split_test) {
            SPLIT_TEST(PREPROCESS.out.processed)
            EMBED(SPLIT_TEST.out.train_val.map { f -> tuple('train', f) })
        } else {
            EMBED(PREPROCESS.out.processed.map { f -> tuple('train', f) })
        }
        embedding_ch = EMBED.out.embedding.map { label, f -> f }
    }

    TRAIN(embedding_ch)
}

workflow predict {

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
    
    if (params.embedding_pt) {
        embedding_ch = Channel.fromPath(params.embedding_pt, checkIfExists: true)
    } else {
        if ((!params.input_tsv)&&(!params.input_json)) {
            error "Please provide --input_tsv <path to raw TSV file> or --input_json <path ot preprocessed JSON>"
        }
        if (params.input_json) {
            processed_ch = Channel.fromPath(params.input_json, checkIfExists: true)
        } else {
            input_ch = Channel.fromPath(params.input_tsv, checkIfExists: true)
            PREPROCESS(input_ch)
            processed_ch = PREPROCESS.out.processed
        }
        EMBED(processed_ch.map { f -> tuple('predict', f) })
        embedding_ch = EMBED.out.embedding.map { label, f -> f }
    }

    PREDICT(embedding_ch, model_ch, threshold_ch)
}

process PREPROCESS {
    tag "preprocess"
    publishDir "${params.cache_dir}/preprocess", mode: 'copy', pattern: "processed.json"
    publishDir "${params.outdir}/train", mode: 'copy', pattern: "preprocess.log"
    debug true

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
    publishDir "${params.cache_dir}/split-test", mode: 'copy', pattern: "processed-{train-val,test}.json"
    publishDir "${params.outdir}/train", mode: 'copy', pattern: "split-test.log"
    debug true

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
    tag "embed-${label}"
    publishDir path: { "${params.cache_dir}/embeddings" }, mode: 'copy', pattern: "embedding-data-*.pt"
    publishDir path: { "${params.outdir}/${label}/embeddings" }, mode: 'copy', pattern: "embed-*.log"
    accelerator 1
    debug true

    input:
    tuple val(label), path(processed_json)

    output:
    tuple val(label), path("embedding-data-${label}.pt"), emit: embedding
    path "embed-${label}.log", emit: log

    script:
    """
    get-embeddings.py \\
        --input ${processed_json} \\
        --output embedding-data-${label}.pt \\
        --batch_size ${params.batch_size_embed} 2>&1 | tee embed-${label}.log
    """
}

process TRAIN {
    tag "train"
    publishDir "${params.outdir}/train/model", mode: 'copy'
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
        --dropout ${params.dropout} \\
        --optimise_metric ${params.optimise_metric} \\
        --classifier ${params.classifier} \\
        --mlp_hidden_size ${params.mlp_hidden_size} 2>&1 | tee train.log
    """
}

process PREDICT {
    tag "predict"
    publishDir "${params.outdir}/predict/", mode: 'copy'
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
        --dropout ${params.dropout} \\
        --threshold ${decision_threshold} \\
        --predictions_out predictions.json \\
        --metrics_out metrics.json \\
        --classifier ${params.classifier} \\
        --mlp_hidden_size ${params.mlp_hidden_size} 2>&1 | tee predict.log
    """
}