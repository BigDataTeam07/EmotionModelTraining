import tensorflow as tf
import numpy as np
from bert import tokenization
from absl import flags
import sys

FLAGS = flags.FLAGS
FLAGS(sys.argv)
VOCAB_FILE = "cased_L-12_H-768_A-12\\vocab.txt"
SAVED_MODEL_DIR = "exported_model\\v3"
EMOTION_FILE = "data\\emotions.txt"
MAX_SEQ_LENGTH = 50
PRED_CUTOFF = 0.05
TOP_K = 3


def convert_sentence_to_features(sentence, tokenizer, max_seq_length):
    tokens = tokenizer.tokenize(sentence)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:max_seq_length - 2]
    tokens = ["[CLS]"] + tokens + ["[SEP]"]

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(tokens)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    return {
        "input_ids": np.array([input_ids], dtype=np.int64),
        "input_mask": np.array([input_mask], dtype=np.int64),
        "segment_ids": np.array([segment_ids], dtype=np.int64)
    }


def create_example(input_ids, input_mask, segment_ids, num_labels):
    features = {
        "input_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=input_ids)),
        "input_mask": tf.train.Feature(int64_list=tf.train.Int64List(value=input_mask)),
        "segment_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=segment_ids)),
        "label_ids": tf.train.Feature(int64_list=tf.train.Int64List(value=[0] * num_labels))
    }
    example = tf.train.Example(features=tf.train.Features(feature=features))
    return example.SerializeToString()


def load_emotions(emotion_file):
    with open(emotion_file, "r", encoding="utf-8") as f:
        emotions = [line.strip() for line in f if line.strip()]
    return emotions


def main():
    # load tokenizer from BERT
    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE, do_lower_case=False)
    # load emotions from file
    emotions = load_emotions(EMOTION_FILE)
    num_labels = len(emotions)
    # load saved model
    predictor = tf.contrib.predictor.from_saved_model(SAVED_MODEL_DIR)
    # sentence to be predicted
    sentence = "I doubt I'll ever own one of those, I'm a bookworm and I love the smell of old books. Thanks for the info, nonetheless."
    print("input text: ", sentence)
    # sentence will be converted to required format
    features = convert_sentence_to_features(sentence, tokenizer, MAX_SEQ_LENGTH)
    # extract the first sample from the batch
    input_ids = features["input_ids"][0].tolist()
    input_mask = features["input_mask"][0].tolist()
    segment_ids = features["segment_ids"][0].tolist()
    # wrap the features into example
    example = create_example(input_ids, input_mask, segment_ids, num_labels)
    # build input dict, key must be "examples", whose value is a list containing serialized strings
    input_dict = {"examples": [example]}
    # start prediction
    predictions = predictor(input_dict)
    # extract probabilities from the prediction result
    probabilities = predictions["probabilities"][0]
    # sort the probabilities in descending order and get top-k results
    sorted_indices = np.argsort(probabilities)[::-1]
    print("predictions (top-%d): " % TOP_K)
    count = 0
    for idx in sorted_indices:
        prob = probabilities[idx]
        if prob < PRED_CUTOFF:
            continue  # if the probability is less than cutoff, skip
        # print the emotion and its probability
        print(f"{emotions[idx]}: {prob:.4f}")
        count += 1
        if count == TOP_K:
            break


if __name__ == "__main__":
    main()
