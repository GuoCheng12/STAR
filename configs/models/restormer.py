_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=2e-4))
test = dict(vis_dir='vis_restormer',visualize=False)

model = dict(type='Restormer',inp_channels=1,out_channels=1,dim=48,num_blocks=[4, 6, 6, 8],
             num_refinement_blocks=4,heads=[1, 2, 4, 8],ffn_expansion_factor=2.66,
             bias=False,LayerNorm_type='BiasFree',dual_pixel_task=False,use_loss='L1',use_attention=False),

dataset = dict(type='SR_dataset',
               batch_size=16,
               num_workers=16,
               root_dir='dataset/x2',
               filenames_file_train='dataset/x2/dataload_filename/train_dataloader.txt',
               filenames_file_eval='dataset/x2/dataload_filename/eval_dataloader.txt',
               
               )
