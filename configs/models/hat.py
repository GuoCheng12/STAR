_base_='../default.py'
seed = 42,
train = dict(max_epoch=100,
             save_ckp_epoch=5,
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
                    steps=[50],
                    steps_unit='epoch',
                )),
             )
test = dict(vis_dir='hatx2',visualize=False)
#              ),
model = dict(type='HAT',
             upscale=2,
                in_chans=1,
                img_size=128,
                window_size=16,
                compress_ratio=3,
                squeeze_factor=30,
                conv_scale=0.01,
                overlap_ratio=0.5,
                img_range=1.,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler='pixelshuffle',
                resi_connection='1conv',use_loss='L1',use_attention=False
             ),
dataset = dict(type='SR_dataset',
               batch_size=6,
               num_workers=4,
               root_dir='dataset/x2',
               filenames_file_train='dataset/x2/dataload_filename/train_dataloader.txt',
               filenames_file_eval='dataset/x2/dataload_filename/eval_dataloader.txt',
               
               )

