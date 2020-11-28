# stPlus
Reference-based enhancement of spatial transcriptomics
## Installation  

stPlus neural network is implemented in [Pytorch](https://pytorch.org/) framework.  
Running stPlus on CUDA is recommended if available.   
	
#### install via pip

	pip install stPlus
	
#### install from GitHub

	git clone git://github.com/xy-chen16/stPlus.git
	cd stPlus
	python setup.py install
    
Installation only requires a few minutes.  

## Quick Start

#### Input

* **spatial_df_file.csv**:   normalized and logarithmized original spatial data
* **scrna_df_file.csv**:  normalized and logarithmized reference scRNA-seq data
* **genes_to_predict.txt**:  spatial genes to be predicted

#### Run

    python stPlus.py -s [spatial_df_file] -i [scrna_df_file] -l[log_file] -g [genes_file] -o [output_file]

#### Output

* **stPlus.log**:  running logs of model
* **save_path_prefix**:  prefix of trained t models with minimal loss
* **stPlus_res.csv**:  predicted spatial data

	

#### Help
Look for more usage of stPlus

	stPlus.py --help 
  
   ```  
  usage: stPlus.py [-h] [--spatial_df_file SPATIAL_DF_FILE] [--scrna_df_file SCRNA_DF_FILE]
                 [--genes_file GENES_FILE] [--output_file OUTPUT_FILE] [--log_file LOG_FILE]
                 [--gpu_id GPU_ID] [--top_k TOP_K] [--t_min T_MIN] [--max_epoch_num MAX_EPOCH_NUM]
                 [--batch_size BATCH_SIZE] [--random_seed RANDOM_SEED] [--save_path_prefix SAVE_PATH_PREFIX]
                 [--data_quality DATA_QUALITY] [--converge_ratio CONVERGE_RATIO]
                 [--learning_rate LEARNING_RATE] [--weight_decay WEIGHT_DECAY] [--verbose VERBOSE]

stPlus: reference-based enhancement of spatial transcriptomics

optional arguments:
  -h, --help            show this help message and exit
  --spatial_df_file SPATIAL_DF_FILE, -s SPATIAL_DF_FILE
                        file path of normalized and logarithmized original spatial data (comma separated CSV
                        file without index)
  --scrna_df_file SCRNA_DF_FILE, -i SCRNA_DF_FILE
                        file path of normalized and logarithmized reference scRNA-seq data (comma separated
                        CSV file without index)
  --genes_file GENES_FILE, -g GENES_FILE
                        file path of spatial genes to be predicted
  --output_file OUTPUT_FILE, -o OUTPUT_FILE
                        file path of predicted spatial data
  --log_file LOG_FILE, -l LOG_FILE
                        file path of running logs
  --gpu_id GPU_ID       ID of GPU to use
  --top_k TOP_K         number of highly variable genes to use
  --t_min T_MIN         number of epochs with minimal loss using to ensemble learning
  --max_epoch_num MAX_EPOCH_NUM
                        maximum number of epochs
  --batch_size BATCH_SIZE
                        batch size for model training
  --random_seed RANDOM_SEED
                        random seed in torch
  --save_path_prefix SAVE_PATH_PREFIX, -m SAVE_PATH_PREFIX
                        prefix of path of trained t models with minimal loss
  --data_quality DATA_QUALITY
                        user-specified or 1 minus the sparsity of scRNA-seq data (default)
  --converge_ratio CONVERGE_RATIO
                        loss converge ratio
  --learning_rate LEARNING_RATE
                        learning rate for model training
  --weight_decay WEIGHT_DECAY
                        weight decay for model training
  --verbose VERBOSE     display the running progress or not   
  
   ```  
Use functions in stPlus packages.

	import stPlus
	from stPlus import *

	
