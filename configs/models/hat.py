_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=2e-4))
test = dict(vis_dir='/fs-computility/ai4sData/zhuangguohang/Astro_SR/vis/vis_restormer_L1',visualize=True)

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
                resi_connection='1conv'
             ),
dataset = dict(type='SR_dataset',
               batch_size=6,
               num_workers=4,
               root_dir='/fs-computility/ai4sData/zhuangguohang/dataset',
               filenames_file_train='/fs-computility/ai4sData/zhuangguohang/dataset/dataload_filename/train_dataloader.txt',
               filenames_file_eval='/fs-computility/ai4sData/zhuangguohang/dataset/dataload_filename/eval_dataloader.txt',
               
               )


               ##   bs--->16  1e-4---->lr        32
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_trainval.sh configs/models/restormer.py