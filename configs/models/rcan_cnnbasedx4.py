_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=2e-4))
test = dict(vis_dir='rcannx4',visualize=True)

#              ),
model = dict(type='RCAN',
            scale=4,
            num_features=64,
            num_rg=10,
            num_rcab=20,  
            reduction=16,   
            use_loss='L1',use_attention=False 
             ),
dataset = dict(type='SR_dataset',
               batch_size=32,
               num_workers=32,
               root_dir='dataset/x4',
               filenames_file_train='dataset/x4/dataload_filename/train_dataloader.txt',
               filenames_file_eval='dataset/x4/dataload_filename/eval_dataloader.txt',
               
               )
