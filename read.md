inputs:  dict_keys([('pose_spatial', -1), ('pose_spatial', 1), 'pose_spatial', 'id', 
('K', 0, 0), ('inv_K', 0, 0), ('K', -1, 0), ('inv_K', -1, 0), ('K', 1, 0),
('inv_K', 1, 0), ('K', 0, 1), ('inv_K', 0, 1), ('K', -1, 1), ('inv_K', -1, 1), 
('K', 1, 1), ('inv_K', 1, 1), ('K', 0, 2), ('inv_K', 0, 2), ('K', -1, 2), 
('inv_K', -1, 2), ('K', 1, 2), ('inv_K', 1, 2), ('K', 0, 3), ('inv_K', 0, 3),
('K', -1, 3), ('inv_K', -1, 3), ('K', 1, 3), ('inv_K', 1, 3), ('color_aug', 0, -1), 
('color', 0, 0), ('color_aug', 0, 0), ('color', 0, 1), ('color_aug', 0, 1), ('color', 0, 2), 
('color_aug', 0, 2), ('color', 0, 3), ('color_aug', 0, 3), ('color', -1, 0), ('color_aug', -1, 0), 
('color', -1, 1), ('color_aug', -1, 1), ('color', -1, 2), ('color_aug', -1, 2), ('color', -1, 3), 
('color_aug', -1, 3), ('color', 1, 0), ('color_aug', 1, 0), ('color', 1, 1), ('color_aug', 1, 1), 
('color', 1, 2), ('color_aug', 1, 2), ('color', 1, 3), ('color_aug', 1, 3)])





features is: <class 'list'> 5 torch.Size([6, 64, 176, 320])
depthdecoder skip is true
after crossCVT is:  torch.Size([6, 64, 176, 320])
depthdecoder skip is true
after crossCVT is:  torch.Size([6, 64, 88, 160])
depthdecoder skip is true
after crossCVT is:  torch.Size([6, 128, 44, 80])
depthdecoder skip is true
after crossCVT is:  torch.Size([6, 256, 22, 40])
depthdecoder skip is true
after crossCVT is:  torch.Size([6, 512, 11, 20])
scales is:  [0, 1, 2, 3]
outputs is:  torch.Size([6, 1, 44, 80])
outputs is:  torch.Size([6, 1, 88, 160])
outputs is:  torch.Size([6, 1, 176, 320])
outputs is:  torch.Size([6, 1, 352, 640])
outputs type is:  <class 'dict'>
self.opt.pose_model_type：  separate_resnet
features is: <class 'list'> 5 torch.Size([6, 64, 176, 320])
depthdecoder skip is true
after crossCVT is:  torch.Size([6, 64, 176, 320])
depthdecoder skip is true
after crossCVT is:  torch.Size([6, 64, 88, 160])
depthdecoder skip is true
after crossCVT is:  torch.Size([6, 128, 44, 80])
depthdecoder skip is true
after crossCVT is:  torch.Size([6, 256, 22, 40])
depthdecoder skip is true
after crossCVT is:  torch.Size([6, 512, 11, 20])
scales is:  [0, 1, 2, 3]
outputs is:  torch.Size([6, 1, 44, 80])
outputs is:  torch.Size([6, 1, 88, 160])
outputs is:  torch.Size([6, 1, 176, 320])
outputs is:  torch.Size([6, 1, 352, 640])











My: level = 2

<class 'dict'> dict_keys(['idx', 'sensor_name', 'filename', 'extrinsics_inv', 'extrinsics', 'mask', ('K', 0), ('inv_K', 0), ('color', 0, 0), ('color_aug', 0, 0), ('K', 1), ('inv_K', 1), ('color', 0, 1), ('color_aug', 0, 1), ('K', 2), ('inv_K', 2), ('color', 0, 2), ('color_aug', 0, 2), ('K', 3), ('inv_K', 3), ('color', 0, 3), ('color_aug', 0, 3), ('color', -1, 0), ('color_aug', -1, 0), ('color', 1, 0), ('color_aug', 1, 0)])
idx:   tensor([12043])
sensor_name:   ['CAM_FRONT']
filename:   ['samples/CAM_FRONT/n008-2018-09-18-14-54-39-0400__CAM_FRONT__1537297717512404.jpg']
extrinsics:   torch.Size([1, 6, 4, 4])
mask:  torch.Size([1, 6, 1, 384, 640])
('K', 0) is: torch.Size([1, 6, 4, 4])
('inv_K', 0) torch.Size([1, 6, 4, 4])
('color', 0, 0) torch.Size([1, 6, 3, 384, 640])
('color', -1, 0)  torch.Size([1, 6, 3, 384, 640])
('color', 1, 0)  torch.Size([1, 6, 3, 384, 640])
('color_aug', 0, 0)   torch.Size([1, 6, 3, 384, 640])
concat image shape is:  torch.Size([1, 6, 3, 384, 640])
after packed image shape is:  torch.Size([6, 3, 384, 640])
e_reault is  <class 'list'>
len(pack_list) 3
every shape after encoder is torch.Size([6, 128, 48, 80]) torch.Size([6, 256, 48, 80]) torch.Size([6, 512, 48, 80])
concated_feature is:  torch.Size([6, 896, 48, 80])
multi_img_feat is:  torch.Size([6, 256, 48, 80])
feats_aggs is:  torch.Size([1, 6, 256, 48, 80])
vox_feat shape is:  torch.Size([1, 64, 200000])
dimg_deat shape is:  <class 'list'> 6 torch.Size([1, 128, 48, 80])




if level = 1

concat image shape is:  torch.Size([1, 6, 3, 384, 640])
after packed image shape is:  torch.Size([6, 3, 384, 640])
e_reault is  <class 'list'>
len(pack_list) 4
every shape after encoder is torch.Size([6, 64, 96, 160]) torch.Size([6, 128, 96, 160]) torch.Size([6, 256, 96, 160])
concated_feature is:  torch.Size([6, 960, 96, 160])
multi_img_feat is:  torch.Size([6, 256, 96, 160])
feats_aggs is:  torch.Size([1, 6, 256, 96, 160])
vox_feat shape is:  torch.Size([1, 64, 200000])
dimg_deat shape is:  <class 'list'> 6 torch.Size([1, 128, 96, 160]


2023 0905 计划

1. 复现surrounddepth源码，使之可以正常loadpertrain
2. 融合两个算法形成第一版本算法
3. 想办法使用position embedding将voxel融合进去
4. 这些都完成后研究loss的影响