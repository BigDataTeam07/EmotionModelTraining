import subprocess

# parameters
export_command = [
    "python", "bert_classifier.py",
    "--do_train=False",
    "--do_eval=False",
    "--do_predict=False",
    "--do_export=True",
    "--multilabel=True",
    "--init_checkpoint=output\\model.ckpt-7500",
    "--output_dir=output",
    "--data_dir=data",
    "--emotion_file=data\\emotions.txt",
    "--vocab_file=cased_L-12_H-768_A-12\\vocab.txt",
    "--bert_config_file=cased_L-12_H-768_A-12\\bert_config.json",
    "--sentiment_file=data\\sentiment_ekman.json",
    "--emotion_correlations=data\\emotion_correlation.tsv",
    "--max_seq_length=50",
    "--do_lower_case=False",
    "--eval_prob_threshold=0.35",  # this value should be determined by sweep_thresholds_eval
    "--sentiment=0.1",
    "--correlation=0.1"
]

print("=== Exporting model to SavedModel ===")
subprocess.run(export_command)
print("Export complete.")
