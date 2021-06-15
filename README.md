# Morphological Analogies using Neural Networks



## Install Instructions
The following installation instruction are designed for command line on Unix systems. Refer to the instructions for Git and Anaconda on your exploitation system for the corresponding instructions.

### Cloning the Repository
Clone the repository on your local machine, using the following command:

```bash
git clone --recurse-submodules https://github.com/AmandineDecker/nn-morpho-analogy.git
```
This will download the repository and its *submodules* (external git repositories, in this case the SIGMORPHON16 dataset).

**[Optional]** To get the GloVe embedding model for German and set it up so that the Dataset class from `data.py`, run:
```bash
mkdir embeddings/glove
grep https://int-emb-glove-de-wiki.s3.eu-central-1.amazonaws.com/vectors.txt embeddings/glove/vectors.txt
```

### Installing the Dependencies

Install Anaconda (or miniconda to save storage space).

Then, create a conda environement (for example `nn-morpho-analogy`) and install the dependencies, using the following commands:

```bash
conda create --name morpho-analogy python=3.9
conda activate morpho-analogy
conda install -y pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch -c=conda-forge
conda install -y -c huggingface transformers
conda install -y numpy scipy pandas matplotlib seaborn
```

### Setting-Up the Japanese Data
The Japanese data is stored as a Sigmorphon2016 data file `japanese-task1-train` at the root of the directory, and should be moved to `sigmorphon2016/data`, the Sigmorphon2016 data folder. This can be done by running the following command:

```bash
mv japanese-task1-train sigmorphon2016/data/
```

There is no test and development set. For the training and evaluation, the file `japanese-task1-train` is split: 70\% of the analogies for the training and 30\% for the evaluation. The split is always the same for reproducibility.

## Usage
For each of the files, it is not necessary to fill the parameters when you run the code, the terminal will ask you to fill them if you did not do it. However, for the parameters in parentheses, if you do not fill them when you launch the code you will not be asked to fill them by the terminal (they are automatically filled).


To train a classifier for a language and the corresponding embedding model, run `python train_cnn_classifier.py --language=<language> --nb_analogies=<number of analogies to use> --epochs=<number of epochs>` (ex: `python train_cnn_classifier.py --language=german --nb_analogies=50000 --epochs=20`).

To evaluate a classifier, run `python evaluate_cnn_classifier.py --language=<language of the classifier> (--epochs=<number of epochs the model was trained for>) --nb_analogies=<number of analogies to use>` (ex: `python evaluate_cnn_classifier.py --language=german --nb_analogies=50000`).

To evaluate all the classifiers on all the languages, run `python evaluate_cnn_classifier_transfer.py --nb_analogies=<number of analogies to use>  (--epochs=<number of epochs the models were trained for>) --mode=<transfer mode>` (ex: `python evaluate_cnn_classifier_transfer.py --nb_analogies=50000 --mode=full). In mode `full` transfer we use the same language for the embedding model and the classifier. In mode `partial` transfer  we use the embedding model corresponding to the data with another classifier. This will output a CSV file containing the accuracy for valid analogies, invalid analogies and valid analogies in base form for all the languages. On the rows we find the values for one model.


To train an analogy solver, run `python train_cnn_solver.py --language=<language of the solver> --nb_analogies=<number of analogies to use> --epochs=<number of epochs>` (ex: `python train_cnn_solver.py --language=german --nb_analogies=50000 --epochs=20`).

To evaluate an analogy solver, run `python evaluate_cnn_solver.py --language=<language of the analogy solver> (--epochs=<number of epochs the model was trained for>) --nb_analogies=<number of analogies to use>` (ex: `python evaluate_cnn_solver.py --language=german --nb_analogies=50000`).

To evaluate an analogy solver with more looseness, run `python evaluate_cnn_solver_extended.py --language=<language of the analogy solver> (--epochs=<number of epochs the model was trained for>) --nb_analogies=<number of analogies to use> --k=<maximum percentage of looseness on the distance to the predicted vector>` (ex: `python evaluate_cnn_solver_extended.py --nb_analogies=50000 --k=5`).


## Files and Folders
- `data.py`: tools to load Sigmorphon2016 datasets, contains the main dataset class `Task1Dataset` and the data augmentation functions `enrich` and `generate_negative`
- `analogy_classif.py`: neural network to classify analogies
- `analogy_reg.py`: neural network to solve analogies
- `cnn_embeddings.py`: neural network to embed words 
- `store_cnn_embeddings.py`: functions to store the embeddings of the train and test set of a given language
- `train_cnn_classifier.py`: file to train the classifier model together with the embedding model for a given language
- `evaluate_cnn_classifier.py`: file to evaluate a classifier with the corresponding embedding model and language
- `evaluate_cnn_classifier_transfer.py`: file to evaluate all the classifiers on other languages with the embedding model corresponding either to the classifier or to the language of the data
- `train_cnn_solver.py`: file to train the analogy solver model for a given language
- `evaluate_cnn_solver.py`: file to evaluate an analogy solver
- `evaluate_cnn_solver_extended.py`: file to evaluate an analogy solver with more looseness, we check if the right tensor is in a given range around the closest vector to the predicted one
- `utils.py`: tools for the different codes
