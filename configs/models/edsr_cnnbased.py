_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=2e-4))
test = dict(vis_dir='vis_edsr',visualize=False)
model = dict(type='EDSR',
             n_resblocks=32, n_feats=256, scale=2,
             res_scale=0.1, 
             n_colors=1, rgb_range=256,use_loss='L1',use_attention=False#gaussian, kaiming, classifier, xavier
             ),
dataset = dict(type='SR_dataset',
               batch_size=32,
               num_workers=32,
               root_dir='dataset/x2',
               filenames_file_train='dataset/x2/dataload_filename/train_dataloader.txt',
               filenames_file_eval='dataset/x2/dataload_filename/eval_dataloader.txt',
               
               )

