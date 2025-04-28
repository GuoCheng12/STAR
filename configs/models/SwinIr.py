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
#test = dict(vis_dir='/home/bingxing2/ailab/zhuangguohang/Astro_SR/Astro_SR/vis_SWIMIR',visualize=True)
model = dict(type='SwinIR',
             img_size=128, in_chans=1, out_chans=1,
             embed_dim=90, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
             window_size=8, mlp_ratio=2., upscale=2, img_range=1.,
             upsampler='pixelshuffle', resi_connection='1conv',initializer='SwinIR', #gaussian, kaiming, classifier, xavier
             ),
# model = dict(type='Restormer',inp_channels=1,out_channels=1,dim=48,num_blocks=[4, 6, 6, 8],
#              num_refinement_blocks=4,heads=[1, 2, 4, 8],ffn_expansion_factor=2.66,
#              bias=True,LayerNorm_type='WithBias',dual_pixel_task=False),
# model = dict(type='HAT',img_size=256, patch_size=1, in_chans=1, embed_dim=96, depths=(6, 6, 6, 6),
#             num_heads=(6, 6, 6, 6),window_size=8,compress_ratio=3,squeeze_factor=30,
#             conv_scale=0.01,overlap_ratio=0.5,mlp_ratio=4., qkv_bias=True,
#             upscale=1,img_range=1.,upsampler='pixelshuffle',resi_connection='1conv',),

# model = dict(type='EDSR'),
# model = dict(type='RCAN'),


dataset = dict(type='SR_dataset',
               batch_size=16,
               num_workers=16,
               root_dir='/ailab/user/wuguocheng/AstroIR/tools/creat_dataset/new_create_dataset/train_patches',
               filenames_file_train='/home/bingxing2/ailab/scxlab0061/Astro_SR/dataload_filename/train_dataloader_gaussian_airy.txt',
               filenames_file_eval='/home/bingxing2/ailab/scxlab0061/Astro_SR/dataload_filename/eval_dataloader_gaussian_airy.txt'
               )
