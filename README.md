# Word-in-Context (WiC) disambiguation

In this project we address Word-in-Context (WiC) disambiguation as a binary classification task, i.e. given two sentences we want to determine whether the indicated polysemous target words have the same meaning or not.
We experimented with a word-level approach (MLP + ReLU) and a sequence encoding one (LSTMs), both on top of GloVe static word embeddings.

For further information, you can read the detailed [report](report.pdf) or take a look at the [presentation slides](presentation.pdf) (pages 2-9).

This project has been developed during the A.Y. 2020-2021 for the [Natural Language Processing](http://naviglinlp.blogspot.com/2021/) course @ Sapienza University of Rome.

## Related projects

- [Word Sense Disambiguation (WSD) for WiC disambiguation](https://github.com/andrea-gasparini/nlp-word-sense-disambiguation-wic-data), experimenting with BERT feature-based and fine-tuning approaches (GlossBERT)
- [Aspect-Based Sentiment Analysis (ABSA)](https://github.com/andrea-gasparini/nlp-aspect-based-sentiment-analysis) using different setups based on 2 stacked BiLSTMs and Attention layers; leveraging PoS, GloVe and BERT (frozen) embeddings

## Authors

- [Andrea Gasparini](https://github.com/andrea-gasparini)

<!--

# NLP-2021: First Homework
This is the first homework of the NLP 2021 course at Sapienza University of Rome.

#### Instructor
* **Roberto Navigli**
	* Webpage: http://wwwusers.di.uniroma1.it/~navigli/

#### Teaching Assistants
* **Cesare Campagnano**
* **Pere-Lluís Huguet Cabot**

#### Course Info
* http://naviglinlp.blogspot.com/

## Requirements

* Ubuntu distribution
	* Either 19.10 or the current LTS are perfectly fine
	* If you do not have it installed, please use a virtual machine (or install it as your secondary OS). Plenty of tutorials online for this part
* [conda](https://docs.conda.io/projects/conda/en/latest/index.html), a package and environment management system particularly used for Python in the ML community

## Notes
Unless otherwise stated, all commands here are expected to be run from the root directory of this project

## Setup Environment

As mentioned in the slides, differently from previous years, this year we will be using Docker to remove any issue pertaining your code runnability. If test.sh runs
on your machine (and you do not edit any uneditable file), it will run on ours as well; we cannot stress enough this point.

Please note that, if it turns out it does not run on our side, and yet you claim it run on yours, the **only explanation** would be that you edited restricted files, 
messing up with the environment reproducibility: regardless of whether or not your code actually runs on your machine, if it does not run on ours, 
you will be failed automatically. **Only edit the allowed files**.

To run *test.sh*, we need to perform two additional steps:
* Install Docker
* Setup a client

For those interested, *test.sh* essentially setups a server exposing your model through a REST Api and then queries this server, evaluating your model.

### Install Docker

```
curl -fsSL get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh
sudo usermod -aG docker $USER
```

Unfortunately, for the latter command to have effect, you need to **logout** and re-login. **Do it** before proceeding. For those who might be
unsure what *logout* means, simply reboot your Ubuntu OS.

### Setup Client

Your model will be exposed through a REST server. In order to call it, we need a client. The client has already been written
(the evaluation script) but it needs some dependecies to run. We will be using conda to create the environment for this client.

```
conda create -n nlp2021-hw1 python=3.7
conda activate nlp2021-hw1
pip install -r requirements.txt
```

## Run

*test.sh* is a simple bash script. To run it:

```
conda activate nlp2021-hw1
bash test.sh data/dev.jsonl
```

Actually, you can replace *data/dev.jsonl* to point to a different file, as far as the target file has the same format.

If you hadn't changed *hw1/stud/model.py* yet when you run test.sh, the scores you just saw describe how a random baseline
behaves. To have *test.sh* evaluate your model, follow the instructions in the slide.
-->
