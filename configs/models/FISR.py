_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999), weight_decay=1e-5))
# test = dict(vis_dir='vis/vis_promptIR_DMP',visualize=False)
test = dict(vis_dir='FISR',visualize=False)
model = dict(type='FISR',inp_channels=1,out_channels=1,dim=48,num_blocks=[4, 6, 6, 8],
             num_refinement_blocks=4,heads=[1, 2, 4, 8],ffn_expansion_factor=2.66,
             bias=True,LayerNorm_type='WithBias',decoder=True,use_loss='L1',use_attention=False),



dataset = dict(type='SR_dataset',
               batch_size=8,
               num_workers=8,
               root_dir='dataset/x2',
               filenames_file_train='dataset/x2/dataload_filename/train_dataloader.txt',
               filenames_file_eval='dataset/x2/dataload_filename/eval_dataloader.txt',
             )
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_trainval.sh configs/models/FISR.py --log_dir log/