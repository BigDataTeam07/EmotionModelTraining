import os
import subprocess

# threshold values for sweep
thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# parameters
base_command = [
    "python", "bert_classifier.py",
    "--do_train=False",
    "--do_eval=True",
    "--calculate_metrics=True",
    "--do_predict=True",
    "--do_export=False",
    "--multilabel=True",
    "--init_checkpoint=checkpoint\\model.ckpt-7500",
    "--output_dir=eval_sweep",
    "--data_dir=data",
    "--emotion_file=data\\emotions.txt",
    "--vocab_file=cased_L-12_H-768_A-12\\vocab.txt",
    "--bert_config_file=cased_L-12_H-768_A-12\\bert_config.json",
    "--sentiment_file=data\\sentiment_ekman.json",
    "--emotion_correlations=data\\emotion_correlation.tsv",
    "--train_fname=train.tsv",
    "--dev_fname=dev.tsv",
    "--test_fname=test.tsv",
    "--max_seq_length=50",
    "--train_batch_size=16",
    "--learning_rate=2e-5",
    "--num_train_epochs=4.0",
    "--correlation=0.1",
    "--sentiment=0.1",
    "--do_lower_case=False"
]

for thres in thresholds:
    print(f"\n=== Evaluating at threshold {thres:.2f} ===")

    thres_output_dir = os.path.join("eval_sweep", f"thres_{int(thres * 100)}")
    os.makedirs(thres_output_dir, exist_ok=True)

    full_command = base_command + [
        f"--eval_prob_threshold={thres}",
        f"--output_dir={thres_output_dir}"
    ]

    subprocess.run(full_command)
