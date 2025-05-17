## Acknowledgements

This project is developed based on the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset and codebase by Google Research.

Original GoEmotions paper:

> Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions", ACL 2020  
> [https://aclanthology.org/2020.acl-main.372](https://aclanthology.org/2020.acl-main.372)  
> [https://github.com/google-research/google-research/tree/master/goemotions](https://github.com/google-research/google-research/tree/master/goemotions)

## Prerequisites
- PC with RTX 20 series GPU (higher will not work, choose CPU instead; lower is not tested)
- Anaconda (recommended)
- Hard Requirements:
  - Python 3.7.X
  - TensorFlow 1.15.5 (both CPU and GPU versions are supported)
  - protobuf 3.20.3
- Rest of the dependencies can be managed using `pip install`
  - numpy
  - pandas
  - absl-py
  - jinja2
  - six
  - etc.
- For GPU support, please install **CUDA Toolkit v10.0** and **cuDNN v7.6.5** and following detailed steps in [TensorFlow GPU installation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/tensorflow-1.14/install.html#tf-install) page.

## Directory structure
- Download and place BERT source codes in the `bert` folder, which will look like this:
    ```
    bert/
    ├── .gitignore
    ├── CONTRIBUTING.md
    ...
    ├── __init__.py
    ├── modeling.py
    ├── optimization.py
    ...
    ├── tokenization.py
    └── tokenization_test.py
    ```
- Download pre-trained BERT model, unzip and place it in the `cased_L-12_H-768_A-12` folder, which will look like this:
    ```
    cased_L-12_H-768_A-12/
    ├── bert_config.json
    ├── bert_model.ckpt.data-00000-of-00001
    ├── bert_model.ckpt.index
    ├── bert_model.ckpt.meta
    └── vocab.txt
    ```

## Build & run commands
- After Successfully installing all the dependencies, run `bert_classifier.py` to train the model
- The parameters are set and free for adjusting in the `flag` area from `bert_classifier.py`
  - the choice of label smoothing and loss function are in **line 388**, which is not explicitly mentioned in the flags
- For better choosing your checkpoint, refer to the `model.ckpt-xxxx.eval_results.txt` files, which contains the evaluation results of the model on the validation set
  - carefully check the **F1@0.30, Accuracy, AUC, Recalls and Loss values**
- If you want to continue training the model, put our provided checkpoint in the `output` folder and run `bert_classifier.py` with set flags
- To determine under which threshold the model is performing best, run `sweep_threshold.py`, be sure to set the same flags as in `bert_classifier.py`
  - choose the best checkpoint through comparing **F1, Precision, Recall, Sentiment F1, Sentiment Recall** values
- Export the model using `export_model.py` with the same flags as in `bert_classifier.py`
  - copy and place the exported model in the `exported_model` folder
- To test your model, run `run_savedmodel.py`
  - You can download our trained model and place it in the `exported_model/v3` folder to test


## Contact
e1285202@u.nus.edu