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
test = dict(vis_dir='/home/bingxing2/ailab/zhuangguohang/Astro_SR/Astro_SR/vis_swimir_L1',visualize=True)
model = dict(type='SwinIR_L1',
             img_size=128, in_chans=1, out_chans=1,
             embed_dim=90, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
             window_size=8, mlp_ratio=2., upscale=2, img_range=1.,
             upsampler='pixelshuffle', resi_connection='1conv',initializer='SwinIR', #gaussian, kaiming, classifier, xavier
             ),


dataset = dict(type='SR_dataset',
               batch_size=16,
               num_workers=16,
               root_dir='/fs-computility/ai4sData/zhuangguohang/dataset',
               filenames_file_train='/fs-computility/ai4sData/zhuangguohang/dataset/dataload_filename/train_dataloader.txt',
               filenames_file_eval='/fs-computility/ai4sData/zhuangguohang/dataset/dataload_filename/eval_dataloader.txt',)
