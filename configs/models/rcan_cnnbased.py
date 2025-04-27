_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=2e-4))
test = dict(vis_dir='/fs-computility/ai4sData/zhuangguohang/Astro_SR/vis/vis_restormer_L1',visualize=True)

#              ),
model = dict(type='RCAN',
            scale=2,
            num_features=64,
            num_rg=10,
            num_rcab=20, 
            reduction=16    
             ),
dataset = dict(type='SR_dataset',
               batch_size=32,
               num_workers=32,
               root_dir='/fs-computility/ai4sData/zhuangguohang/dataset',
               filenames_file_train='/fs-computility/ai4sData/zhuangguohang/dataset/dataload_filename/train_dataloader.txt',
               filenames_file_eval='/fs-computility/ai4sData/zhuangguohang/dataset/dataload_filename/eval_dataloader.txt',
               
               )


               ##   bs--->16  1e-4---->lr        32
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_trainval.sh configs/models/restormer.py