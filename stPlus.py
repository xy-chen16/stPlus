stPlus.py
a
few
seconds
ago
Python
File
Edit
View
Language
r

parser.add_argument('--top_k', default=3000, type=int, help='number of highly variable genes to use')
parser.add_argument('--t_min', default=5, type=int,
                    help='number of epochs with minimal loss using to ensemble learning')
parser.add_argument('--max_epoch_num', default=10000, type=int, help='maximum number of epochs')
parser.add_argument('--batch_size', default=512, type=int, help='batch size for model training')
parser.add_argument('--random_seed', default=None, type=int, help='random seed in torch')

parser.add_argument('--save_path_prefix', '-m', type=str, default='./model',
                    help='prefix of path of trained t models with minimal loss')

parser.add_argument('--data_quality', type=float, default=None,
                    help='user-specified or 1 minus the sparsity of scRNA-seq data (default)')
parser.add_argument('--converge_ratio', type=float, default=0.004, help='loss converge ratio')
parser.add_argument('--learning_rate', type=float, default=None, help='learning rate for model training')
parser.add_argument('--weight_decay', type=float, default=0.0002, help='weight decay for model training')

parser.add_argument('--verbose', type=bool, default=True, help='display the running progress or not')

# Get the paramenters from inputs
args = parser.parse_args()
spatial_df_file = args.spatial_df_file
scrna_df_file = args.scrna_df_file
genes_file = args.genes_file
output_file = args.output_file
log_file = args.log_file
top_k = args.top_k
t_min = args.t_min
max_epoch_num = args.max_epoch_num
batch_size = args.batch_size
random_seed = args.random_seed
save_path_prefix = args.save_path_prefix
data_quality = args.data_quality
converge_ratio = args.converge_ratio
learning_rate = args.learning_rate
weight_decay = args.weight_decay
verbose = args.verbose
if ((spatial_df_file == None) or (scrna_df_file == None) or (genes_file == None) or (output_file == None)):
    sys.exit(
        'Usage: python stPlus.py --spatial_df_file [spatial_df_file] --scrna_df_file [scrna_df_file] --genes_file [genes_file] --output_file [output_file]')
# Loading data
print('Loading data')
spatial_df = pd.read_csv(spatial_df_file)
scrna_df = pd.read_csv(scrna_df_file)
genes_to_predict = pd.read_csv(genes_file, header=None).iloc[:, 0].values.tolist()
print('Predicted results will be saved in: %s' % (output_file))
print('Running logs will be saved in: %s' % (log_file))
# Train the model and save the results
if not os.path.exists(save_path_prefix):
    os.makedirs(save_path_prefix)
stPlus_res = stPlus(spatial_df=spatial_df, scrna_df=scrna_df, genes_to_predict=genes_to_predict,
                    save_path_prefix=save_path_prefix, top_k=top_k, t_min=t_min, data_quality=data_quality,
                    random_seed=random_seed, verbose=verbose, converge_ratio=converge_ratio,
                    max_epoch_num=max_epoch_num, batch_size=batch_size, learning_rate=learning_rate,
                    weight_decay=weight_decay)

if output_file is not None:
    stPlus_res.to_csv(output_file, index=False)

if log_file is not None:
    log_file = open(log_file, 'w')
    log_file.write('{}'.format(args))
    log_file.close()
