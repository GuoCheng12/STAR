_base_='../default.py'
seed = 42,
train = dict(optimizer=dict(lr=2e-4))
test = dict(vis_dir='promptIR',visualize=True)
model = dict(type='PromptIR',inp_channels=1,out_channels=1,dim=48,num_blocks=[4, 6, 6, 8],
             num_refinement_blocks=4,heads=[1, 2, 4, 8],ffn_expansion_factor=2.66,
             bias=True,LayerNorm_type='WithBias',decoder=True,use_loss='L1',use_attention=False),



dataset = dict(type='SR_dataset',
               batch_size=32,
               num_workers=32,
               root_dir='dataset/x2',
               filenames_file_train='dataset/x2/dataload_filename/train_dataloader.txt',
               filenames_file_eval='dataset/x2/dataload_filename/eval_dataloader1.txt',
               
               )