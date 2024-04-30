# ROCES
Robust Class Expression Synthesis in Description Logics via Iterative Sampling


## Installation

Make sure Anaconda3 is installed in your working environment then run the following to install all required librairies for ROCES:
```
conda create -n roces python==3.9.0 --y && conda activate roces && pip install -r requirements.txt
```

A conda environment (roces) will be created. Next activate the environment:

```
conda activate roces
```

Also install Ontolearn: 

``` 
git clone --branch 0.5.4 --depth 1 https://github.com/dice-group/Ontolearn.git
```
then

``` 
cd Ontolearn && pip install -e .
```

- To run CELOE from DL-Learner, first install Java 8+ and Maven 3.6.3+


## Dependencies
1. python 3.9.0
2. numpy 1.26.2
3. pandas 2.1.3
4. tqdm 4.66.1
5. scikit-learn 1.3.2
6. torch 2.1.1
7. transformers 4.35.2
8. matplotlib 3.8.2
9. seaborn 0.13.0


## Hardware:
- Ran on Debian GNU/Linux 12 AMD EPYC 9334 32-Core Processor @ 3.91GHz (64 CPUs), 1xNvidia GPU A100 80GB, 1TB RAM
- ROCES requires at least 12GB RAM to run on the benchmark datasets

## Reproducing the reported results

- First download datasets and pretrained models from [here](https://drive.google.com/file/d/1j5nlaCPugHAF5ggFO0cBHL7L2YgUkAcS/view?usp=sharing). Extract the Zip file into the top directory ROCES and make sure the resulting folder is named `datasets`.


### ROCES (Ours)


*Open a terminal in ROCES/*

- To reproduce training curves and obtain the pretrained models we provided, run ``` python train.py ```; the default dataset is ` carcinogenesis `. To train on a different dataset, use ```--kbs ```, e.g., `--kbs vicodi`. Run `python train.py -h` to view all available options.

- To reproduce the main results (Table 2, Figures 1 and 2), run: ``` python run_nces2_and_roces_incremental.py ```. Specify the approach `--approach roces`

- To reproduce results on learning problems with full example sets (Table 2), run: ``` python run_roces.py ```. Use -h for more options.

- For the results reported in Table 3, please refer to the notebook roces_success_rate.ipynb


### NCES2 (Kouagou et al. 2023)

*Open a terminal in ROCES/*
- To reproduce the main results (Table 2, Figures 2 and 3), run: ``` python run_nces2_and_roces_incremental.py ```. Specify the approach `--approach nces2`

- Results on learning problems with full example sets (Table 2) can be found in the original paper https://link.springer.com/chapter/10.1007/978-3-031-43421-1_12, Table 4


### EvoLearner (Heindorf et al. 2022)

*Open a terminal and navigate into evolearner/* ``` cd ROCES/evolearner/ ```

- Run `python run_evolearner_incremental.py --kb `. Use options to, e.g., select the knowledge base or save results.

- Results on learning problems with full example sets (Table 3) can be found in the paper https://link.springer.com/chapter/10.1007/978-3-031-43421-1_12, Table 4


### DL-Learner (Lehmann et al. 2011)

*Open a terminal and navigate into dllearner/* ``` cd ROCES/dllearner ```

- Reproduce CELOE concept learning results: ``` python run_dllearner_incremental.py --kb ```

- Results on learning problems with full example sets (Table 3) can be found in the paper https://link.springer.com/chapter/10.1007/978-3-031-43421-1_12, Table 4


*Remark: Throughout this documentation, --kbs/--kb specifies one of carcinogenesis, mutagenesis, semantic_bible, or vicodi*

## Bring your own data

To train ROCES on a new knowledge base, create a new folder under datasets and add the OWL format of the knowledge base in the folder. Make sure the owl file has the same name as the folder you created. Follow the 3 steps below to train ROCES on your knowledge base.

- (1) Generating training data for ROCES: `cd generators/` then ` python generate_data.py --kbs your_folder_name `. Use -h for more options. For example, use `--num_rand_samples 500` combined with `--refinement_expressivity 0.6` to increase the amount of training data.

- (2) Convert knowledge base to knowledge graph: ```cd generators ``` then ``` python kb_to_kg.py --kbs your_folder_name ```

- (3) Training ROCES on your data: `cd ROCES/ ` then ` python train.py --kbs your_folder_name `. Use -h to see more options for training, e.g., `--batch_size` or `--epochs`

