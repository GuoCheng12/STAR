_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=2e-4))
test = dict(vis_dir='/home/bingxing2/ailab/zhuangguohang/Astro_SR/Astro_SR/vis_promptIR',visualize=True)
# model = dict(type='SwinIR',
#              img_size=128, patch_size=1, in_chans=1, out_chans=1,
#              embed_dim=90, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
#              window_size=8, mlp_ratio=2., upscale=2, img_range=1.,
#              upsampler='pixelshuffledirect', resi_connection='1conv'
#              ),
model = dict(type='PromptIR_L1',inp_channels=1,out_channels=1,dim=48,num_blocks=[4, 6, 6, 8],
             num_refinement_blocks=4,heads=[1, 2, 4, 8],ffn_expansion_factor=2.66,
             bias=True,LayerNorm_type='WithBias',decoder=True),



dataset = dict(type='SR_dataset',
               batch_size=16,
               num_workers=16,
               root_dir='/fs-computility/ai4sData/zhuangguohang/dataset',
               filenames_file_train='/fs-computility/ai4sData/zhuangguohang/dataset/dataload_filename/train_dataloader.txt',
               filenames_file_eval='/fs-computility/ai4sData/zhuangguohang/dataset/dataload_filename/eval_dataloader.txt',
               
               )
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_trainval.sh configs/models/restormer.py