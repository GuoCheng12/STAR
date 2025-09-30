_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=2e-4))
test = dict(vis_dir='bilinearx2',visualize=False)
model = dict(type='Bilinear',         
            scale_factor=2, 
            mode = 'bilinear',),#bicubic ,  'bilinear'


dataset = dict(type='SR_dataset',
               batch_size=24,
               num_workers=16,
               root_dir='dataset/x2',
               filenames_file_train='dataset/x2/dataload_filename/train_dataloader.txt',
               filenames_file_eval='dataset/x2/dataload_filename/eval_dataloader.txt',
               
               )