_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=4e-4))
test = dict(vis_dir='esrgan1',visualize=False)

model = dict(type='ESRGAN',         
                num_in_ch=1,
                num_feat =64,
                num_out_ch=1,
                num_block=23,
                use_loss='L1',
                num_grow_ch=32,
                scale=2,
                use_attention=False,
                stage = 'two',
                checkpoints = './best_epoch.pt'),#bicubic ,  'bilinear'

dataset = dict(type='SR_dataset',
               batch_size=32,
               num_workers=32,
               root_dir='dataset/x2',
               filenames_file_train='dataset/x2/dataload_filename/train_dataloader.txt',
               filenames_file_eval='dataset/x2/dataload_filename/eval_dataloader.txt',
               
               )
