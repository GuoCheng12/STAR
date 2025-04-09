seed = 42,
train = dict(max_epoch=10,
            save_ckp_epoch=1,
            eval_epoch=1,
            display_iter=10,
            grad_clip=None, #dict(max_norm=20, norm_type=2),
            optimizer=dict(
                type='Adam',
                lr=1e-3),   
            scheduler=dict(
                 warm_up = dict(
                      type='linear',
                      ratio=1.0,
                      step_type='iter',
                      bound=1, 
                      bound_unit='epoch',
                 ),
                 lr_decay=dict(
                      type='cos',  #cos, step
                      step_type='iter',
                      # decay_ratio=0.1,  # step decay parameters
                      steps=[40],
                      steps_unit='epoch')      
            )),

# train = dict(max_epoch=400,
#                save_ckp_epoch=10,
#                eval_epoch=1,
#                display_iter=1,
#                grad_clip=None,
#                optimizer=dict(
#                     type='SGD',
#                     lr=5e-3,
#                     weight_decay=1e-4,
#                     momentum=0.9),
#                scheduler=dict(
#                  warm_up = dict(
#                       type='linear',
#                       ratio=0.0,
#                       step_type='iter',
#                       bound=1, 
#                       bound_unit='epoch',
#                  ),
#                  lr_decay=dict(
#                       type='cos',  #cos, step
#                       step_type='iter',
#                       decay_ratio=0.1,  # step decay parameters
#                       steps=[300,360],
#                       steps_unit='epoch'))   
# ),
test = dict(vis_dir='/home/bingxing2/ailab/zhuangguohang/Astro_SR/Astro_SR/vis/'),

# model = dict(type='Simple_baseline',
#              n_channels=2, 
#              initializer='gaussian',  #gaussian, kaiming, classifier, xavier
#              bilinear=False,  #bilinear cannot reproduce
#              losses=dict(
#                 cls_loss=dict(type='curriculum_focal_loss_heatmap', 
#                      alpha=2, 
#                      gamma=4,
#                      weight=1.0))
#              ),  
# dataset=dict(type='FAST_detect',
#              root_dir='/mnt/workspace/Datasets/astronomy/fast_pz', 
#              disk_root='./data',
#              backend='disk',
#              cepth_config='/mnt/petrelfs/luyan/petreloss_new.conf',
#              repeat_num=10,
#              batch_size=16, 
#              num_workers=8,
#              voxel_size=dict(ra=256,
#                              dec=32,
#                              freq=512),   # 3d res for reconstruction
#              )  
model = dict(type='Simple_baseline',
             #n_channels=2, 
             initializer='gaussian',  #gaussian, kaiming, classifier, xavier
             bilinear=False,
               #bilinear cannot reproduce
             losses=dict(
                     L1_loss=dict(type='L1_loss', 
                                        weight=1.0
                                        ))
             ),
dataset=dict(type='SR_astro',
          batch_size=8,
          num_worker=6,
               )

          