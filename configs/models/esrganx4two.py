_base_='../default.py'
seed = 42,
# train = dict(optimizer=dict(lr=2e-4))
train = dict(max_epoch=100,
             save_ckp_epoch=2,
             eval_epoch=1,
             display_iter=10,
             grad_clip=None,
            optimizer=dict(type='Adam', lr=2e-4, betas=(0.9, 0.999), weight_decay=0),
             scheduler=dict(
                warm_up = dict(
                type='linear',
                ratio=0.01, 
                step_type='iter',
                bound=1, 
                bound_unit='epoch'
             ),
                lr_decay=dict(
                    type='cos',
                    step_type='epoch',
                    steps=[25],
                    steps_unit='epoch',
                )),
             )
test = dict(vis_dir='..',visualize=False)

model = dict(type='ESRGAN',         
                num_in_ch=1,
                num_feat =64,
                num_out_ch=1,
                num_block=23,
                use_loss='L1',
                num_grow_ch=32,
                scale=4,
                use_attention=False,
                stage = 'two',
                checkpoints='./epoch_xx.pth'),#bicubic ,  'bilinear'



dataset = dict(type='SR_dataset',
               batch_size=32,
               num_workers=32,
               root_dir='dataset/x4',
               filenames_file_train='dataset/x4/dataload_filename/train_dataloader.txt',
               filenames_file_eval='dataset/x4/dataload_filename/eval_dataloader.txt',
               
               )

