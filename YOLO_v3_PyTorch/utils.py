import torch 
import numpy as np


def bbox_iou(box1, box2):
    
    # box数据格式：x1 y1（左上角） x2 y2（右下角）
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:,0], box1[:,1], box1[:,2], box1[:,3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:,0], box2[:,1], box2[:,2], box2[:,3]
    
    # 重合区域左上角坐标：两个box的x1,y1中较大者
    inter_x1 = torch.max(b1_x1, b2_x1)
    inter_y1 = torch.max(b1_y1, b2_y1)
    
    # 重合区域右下角坐标：两个box的x2,y2中较小者
    inter_x2 = torch.min(b1_x2, b2_x2)
    inter_y2 = torch.min(b1_y2, b2_y2)
    
    # torch.clamp：如果参数小于min，则输出min
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                 torch.clamp(inter_y2 - inter_y1, min=0)
        
    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
        
    iou = inter_area / (b1_area + b2_area - inter_area)
    return iou


def predict_transform(pred, anchors, input_dim, num_classes):
    
    # 是否支持CUDA
    CUDA = torch.cuda.is_available()
    
    # pred: 来自卷积神经网络的输出，第0个维度为batch size
    batch_size = pred.shape[0]
    stride = input_dim // pred.shape[2]
    num_grids = pred.shape[2]

    # BoundingBox属性向量的长度
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)
        
    # [1, 75, num_grids, num_grids] -> [1, num_grids * num_grids * num_anchors, 25]
    # pred: 每行对应一个Bounding Box属性向量
    pred = pred.view(batch_size, bbox_attrs * num_anchors, num_grids * num_grids)
    pred = pred.transpose(1,2).contiguous()
    pred = pred.view(batch_size, num_grids * num_grids * num_anchors, bbox_attrs)
    
    pred[:,:,0] = torch.sigmoid(pred[:,:,0])  # center_x
    pred[:,:,1] = torch.sigmoid(pred[:,:,1])  # center_y
    pred[:,:,4] = torch.sigmoid(pred[:,:,4])  # object confidence
        
    # 计算BBox相对Grid的偏移
    grid = np.arange(num_grids)
    a, b = np.meshgrid(grid, grid)

    x_offset = torch.FloatTensor(a).view(-1, 1)
    y_offset = torch.FloatTensor(b).view(-1, 1)

    # 如果支持CUDA，则转换为GPU并行变量
    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()
        pred = pred.cuda()
        
    x_y_offset = torch.cat((x_offset, y_offset), 1).repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    pred[:,:,:2] += x_y_offset
    
    # 根据stride变换Anchors尺寸
    anchors = [(a[0]/stride, a[1]/stride) for a in anchors]
    
    anchors = torch.FloatTensor(anchors) 
    
    if CUDA:
        anchors = anchors.cuda()
    
    anchors = anchors.repeat(num_grids * num_grids, 1).unsqueeze(0)
    
    # 根据Anchors尺寸变换预测输出
    pred[:,:,2:4] = torch.exp(pred[:,:,2:4]) * anchors
    pred[:,:,5:] = torch.sigmoid(pred[:,:,5:])

    # x, y, w, h：还原到416x416尺度
    pred[:,:,:4] *= stride
    
    return pred


def bbox_nms(pred, obj_conf=0.5, nms_thres=0.4):

    # pred[:,:,4]：object confidence
    # .float()：布尔变量 -> 浮点数
    # .unsqueeze(2)：增加维度，与pred形状一致
    # [1, 10647] -> [1, 10647, 1]
    conf_mask = (pred[:,:,4] > obj_conf).float().unsqueeze(2)
    
    # 与mask相乘，大于conf的保留，小于conf的变为0
    pred = pred * conf_mask

    box_corner = pred.new(pred.shape)
    # BBox左上角x坐标 = BBox中心x坐标 - BBox宽度的一半
    box_corner[:,:,0] = pred[:,:,0] - pred[:,:,2] / 2
    # BBox左上角y坐标 = BBox中心y坐标 - BBox高度的一半
    box_corner[:,:,1] = pred[:,:,1] - pred[:,:,3] / 2
    # BBox右下角x坐标 = BBox中心x坐标 + BBox宽度的一半
    box_corner[:,:,2] = pred[:,:,0] + pred[:,:,2] / 2
    # BBox右下角y坐标 = BBox中心y坐标 + BBox高度的一半
    box_corner[:,:,3] = pred[:,:,1] + pred[:,:,3] / 2
    pred[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = pred.shape[0]
    write = 0

    # 对Batch中的样本循环
    for i in range(batch_size):
        
        bbox_pred = pred[i]
        
        # torch.max(<Tensor>, 1): 返回第1维上的最大值，即按列压缩，取每一行的最大值
        # max_conf：返回每行（grid）的最大值
        # max_conf_score：返回每行最大值出现的位置，也就是20个类别中概率最大的
        max_conf, max_conf_id = torch.max(bbox_pred[:,5:], 1)
        
        # 将因为.max()压缩的维度填充回去，保持与pred一致
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_id = max_conf_id.float().unsqueeze(1)
        
        bbox_pred = torch.cat((bbox_pred[:,:5], max_conf, max_conf_id), 1)
        # 查找object confidence非零（即原始输出大于0.5）的grid编号
        non_zero_id = torch.nonzero(bbox_pred[:,4]).squeeze()
        
        # 筛选object confidence非零的Bouding Box
        bbox_det = bbox_pred[non_zero_id]
        
        # 如果没有非零confidence（图里没检测到东西）
        if bbox_det.shape[0] == 0:
            continue
        
        # image_det[:,-1]: -1列为最大概率对应类别（0-19）
        det_classes = torch.unique(bbox_det[:,-1])
        
        # 对数据集中的类别循环
        for cls in det_classes:
            
            # cls_mask: 筛选最大概率的类别是当前cls的BBox
            cls_mask = bbox_det * (bbox_det[:,-1] == cls).float().unsqueeze(1)
            cls_mask_id = torch.nonzero(cls_mask[:,-2]).squeeze()
            # bbox_det -> bbox_det_class: 针对当前类别的检测
            bbox_det_class = bbox_det[cls_mask_id].view(-1,7)
                                    
            # 在当前类别cls中，按照confidence高到低排序
            conf_sort_id = torch.sort(bbox_det_class[:,4], descending=True)[1]
            # 对当前类别cls的检测结果重新排序
            bbox_det_class = bbox_det_class[conf_sort_id]
            
            num_bboxes = bbox_det_class.shape[0]
            
            # 对BoundingBoxes循环
            for j in range(num_bboxes-1):
                
                # 计算第j个BBox，与之后所有BBox的IoU
                ious = bbox_iou(bbox_det_class[j].unsqueeze(0), bbox_det_class[j+1:])
                
                # 如果存在第j个之后的BBox，与当前BBox的IoU过高，则过滤为0
                # 因为bbox_det_class已经按照confidence高到低排序，所以扔掉后面的
                iou_mask = (ious < nms_thres).float().unsqueeze(1)
                bbox_det_class[j+1:] *= iou_mask
                
            non_zero_id = torch.nonzero(bbox_det_class[:,4]).squeeze()
            bbox_det_class = bbox_det_class[non_zero_id].view(-1,7)  
            
            # 行数：经过NMS的BBox数量，列数：1，以当前sample id填充
            sample_id = bbox_det_class.new(bbox_det_class.shape[0], 1).fill_(i)
            detections = torch.cat((sample_id, bbox_det_class), 1)
            
            if not write:
                output = detections
                write = 1
                
            else:
                output = torch.cat((output, detections))

    return output


def bbox_transform(output, ori_img, img_size):
    
    # torch.clamp:如果坐标位置超出图片范围，则改为图片边缘
    # 按比例缩放BBox
    output_ = output.clone()
    output_[:, 1:5] = torch.clamp(output[:, 1:5], 0, float(img_size))
    output_  *=  (ori_img.shape[0] / img_size)
    
    # 获取BBox的左上-右下角坐标
    # .int(): 绘制图片需要整数型坐标点
    pt1 = output_[:,1:3].data.int().cpu().numpy()
    pt2 = output_[:,3:5].data.int().cpu().numpy()
    
    pt1 = [tuple(p) for p in pt1]
    pt2 = [tuple(p) for p in pt2]
    
    return pt1, pt2
