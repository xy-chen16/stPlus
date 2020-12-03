# stPlus
stPlus is a reference-based method for the enhancement of spatial transcriptomics. 
Leveraging the holistic information in reference scRNA-seq data but not limited to the genes shared with spatial data, 
stPlus performs non-linear embedding for cells in both datasets and effectively predicts unmeasured spatial gene expression.

## Installation  
Anaconda users can first create a new Python environment and activate it via (this is unnecessary if your Python environment is managed in other ways)
```
conda create python=3.8 -n stPlus
conda activate stPlus
```

stPlus is implemented based on the [Pytorch](https://pytorch.org/) framework. Running stPlus on CUDA is recommended if available.
For the reproduction of results, we install Pytorch via

```
pip install torch==1.6.0+cu92 torchvision==0.7.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

You can also access the installation command [here](https://pytorch.org/) for specific operating system, CUDA version, and Pytorch version.

stPlus is available on pypi [here](https://pypi.org/project/stPlus/) and can be installed via

```
pip install stPlus
```

You can also install stPlus from GitHub via
```
git clone git://github.com/xy-chen16/stPlus.git
cd stPlus
python setup.py install
```
The dependencies will be installed along with stPlus.


## Quick Start

### Input

* **spatial_df**:       normalized and logarithmized original spatial data
* **scrna_df**:         normalized and logarithmized reference scRNA-seq data
* **genes_to_predict**: spatial genes to be predicted

### Output

* **stPlus_res**:       predicted spatial transcriptomics data

### For calling stPlus programmatically
```python
	import pandas as pd
	from stPlus import *
	# Load the normalized and logarithmized spatial and scRNA-seq data, and the genes to predict
	# The data can be accessed via: 
	# 	git clone git://github.com/xy-chen16/stPlus.git
	# 	cd stPlus
	# 	tar -zxvf data.tar.gz
	spatial_df_file = './data/osmFISH_df.csv'
	scrna_df_file   = './data/Zeisel_df.csv'
	genes_file = './data/genes_to_predict.txt'
	spatial_df = pd.read_csv(spatial_df_file) # (cell by gene pandas dataframe)
	scrna_df   = pd.read_csv(scrna_df_file)   # (cell by gene pandas dataframe)
	genes_to_predict = pd.read_csv(genes_file, header=None).iloc[:,0].values # 1D numpy array
	# Run stPlus
	stPlus_res = stPlus(spatial_df, scrna_df, genes_to_predict)
```
    
stPlus can also be seamlessly integrated with [Scanpy](https://scanpy.readthedocs.io/en/stable/), a widely-used Python library for the analysis of single-cell data.
```python
	import scanpy as sc
	# Load the spatial and scRNA-seq data as Scanpy objects (spatial_sc and scrna_sc)
	# Normalize and logarithmize if the data contains raw counts
	sc.pp.normalize_total(spatial_sc)
	sc.pp.log1p(spatial_sc)
	sc.pp.normalize_total(scrna_sc)
	sc.pp.log1p(scrna_sc)
	# Transfer to DataFrames
	spatial_df = pd.DataFrame(spatial_sc.X, columns=spatial_sc.var.index.values)
	scrna_df   = pd.DataFrame(scrna_sc.X, columns=scrna_sc.var.index.values)
	# Run stPlus
	stPlus_res = stPlus(spatial_df, scrna_df, genes_to_predict)
```
### Documentation notebook 

We also provide a [quick-start notebook](https://github.com/xy-chen16/stPlus/demo.ipynb) which describes the fundamentals in detail and reproduces the results of stPlus.

### For calling stPlus with Bash commands
```
git clone git://github.com/xy-chen16/stPlus.git
cd stPlus
tar -zxvf data.tar.gz
python stPlus.py --spatial_df_file data/osmFISH_df.csv  --scrna_df_file data/Zeisel_df.csv \
		 --genes_file data/genes_to_predict.txt --output_file stPlus_res.csv
```
Look for more usage of stPlus iva

```
python stPlus.py --help
```
  
```  
usage: stPlus.py [--help] [--spatial_df_file SPATIAL_DF_FILE] [--scrna_df_file SCRNA_DF_FILE]
                 [--genes_file GENES_FILE] [--output_file OUTPUT_FILE] [--log_file LOG_FILE]
                 [--gpu_id GPU_ID] [--top_k TOP_K] [--t_min T_MIN] [--max_epoch_num MAX_EPOCH_NUM]
                 [--batch_size BATCH_SIZE] [--random_seed RANDOM_SEED] [--save_path_prefix SAVE_PATH_PREFIX]
                 [--data_quality DATA_QUALITY] [--converge_ratio CONVERGE_RATIO]
                 [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--verbose VERBOSE]

stPlus: reference-based enhancement of spatial transcriptomics

optional arguments:
	--help, -h
		show the help message and exit
	--spatial_df_file SPATIAL_DF_FILE, -s SPATIAL_DF_FILE
		file path of normalized and logarithmized original spatial data (comma separated CSV
		file without index)
	--scrna_df_file SCRNA_DF_FILE, -r SCRNA_DF_FILE
		file path of normalized and logarithmized reference scRNA-seq data (comma separated
		CSV file without index)
	--genes_file GENES_FILE, -g GENES_FILE
		file path of spatial genes to be predicted
	--output_file OUTPUT_FILE, -o OUTPUT_FILE
		file path of predicted spatial data
	--log_file LOG_FILE,  LOG_FILE
		file path of running logs
	--gpu_id GPU_ID
		ID of GPU to use
	--top_k TOP_K
		number of highly variable genes to use
	--t_min T_MIN
		number of epochs with minimal loss using to ensemble learning
	--max_epoch_num MAX_EPOCH_NUM
		maximum number of epochs
	--batch_size BATCH_SIZE
		batch size for model training
	--random_seed RANDOM_SEED
		random seed in torch
	--save_path_prefix SAVE_PATH_PREFIX
		prefix of path of trained t models with minimal loss
	--data_quality DATA_QUALITY
		user-specified or 1 minus the sparsity of scRNA-seq data (default)
	--converge_ratio CONVERGE_RATIO
		loss converge ratio
	--learning_rate LEARNING_RATE
		learning rate for model training
	--weight_decay WEIGHT_DECAY
		weight decay for model training
	--verbose VERBOSE     
		display the running progress or not   
```  
