seed = 42,
train = dict(max_epoch=1000,
             save_ckp_epoch=5,
             eval_epoch=1,
             display_iter=10,
             grad_clip=None,
            optimizer=dict(type='Adam', lr=1e-4, betas=(0.9, 0.999), weight_decay=0),
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
                    steps=[0],
                    steps_unit='epoch',
                )),
             )

test = dict(vis_dir='vis/'),


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
          batch_size=16,
          num_worker=6,
               )

          