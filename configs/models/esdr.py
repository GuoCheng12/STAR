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
model = dict(type='EDSR',
             n_resblocks=32, n_feats=256, scale=2,
             res_scale=0.1, 
             n_colors=1, rgb_range=256#gaussian, kaiming, classifier, xavier
             ),

dataset = dict(type='SR_dataset',
               batch_size=24,
               num_workers=16,
               root_dir='/ailab/user/wuguocheng/AstroIR/tools/creat_dataset/new_create_dataset/train_patches',
               filenames_file_train='/home/bingxing2/ailab/scxlab0061/Astro_SR/dataload_filename/train_dataloader_gaussian_airy.txt',
               filenames_file_eval='/home/bingxing2/ailab/scxlab0061/Astro_SR/dataload_filename/eval_dataloader_gaussian_airy.txt'
               )