_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=2e-4))
test = dict(vis_dir='/home/bingxing2/ailab/zhuangguohang/Astro_SR/Astro_SR/bilinear',visualize=False)

model = dict(type='Bilinear',         
            scale_factor=2, 
            mode = 'bilinear',),#bicubic ,  'bilinear'



dataset = dict(type='SR_dataset',
               batch_size=16,
               num_workers=16,
               root_dir='/fs-computility/ai4sData/zhuangguohang/dataset',
               filenames_file_train='/fs-computility/ai4sData/zhuangguohang/dataset/dataload_filename/train_dataloader.txt',
               filenames_file_eval='/fs-computility/ai4sData/zhuangguohang/dataset/dataload_filename/eval_dataloader.txt',
               
               )
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash tools/dist_trainval.sh configs/models/restormer.py