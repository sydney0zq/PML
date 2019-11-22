import os
import sys
import cv2
from PIL import Image
import numpy as np

label_colours = [(0,0,0) # 0=background
                ,(0,0,128)]

def calcIoU(gt, pred, obj_n): 
    assert(gt.shape == pred.shape)
    ious = np.zeros((obj_n), dtype=np.float32)
    for obj_id in range(1, obj_n+1):
        gt_mask = gt == obj_id
        pred_mask = pred == obj_id
        inter = gt_mask & pred_mask
        union = gt_mask | pred_mask
        if union.sum() == 0:
            ious[obj_id-1] = 1
        else:
            ious[obj_id-1] = float(inter.sum()) / union.sum()
    return ious


def compute_mIOU(data_path, pred_path, dataset_version, dataset_split, vis_path=None, logger=None):
    logger = logger if logger is not None else logging
    logger.info("Start to do final mIOU computation...")
    listFile = '%s/ImageSets/%s/%s.txt' % (data_path, dataset_version, dataset_split)
    gt_path = os.path.join(data_path, 'Annotations', '480p')
    with open(listFile, 'r') as f:
        fds = [line.strip() for line in f]
    im_num = 0
    iou =[]
    seq_n = 0
    sample_n = 0
    subfd_names = []
    for i, fd in enumerate(fds):
        file_list = os.listdir(os.path.join(gt_path, fd))
        im_list = [name for name in file_list if len(name) > 4 and name[-4:]=='.png']
        im_list = sorted(im_list)
        im_list = im_list[1:-1] # remove first and last image
        pred_list = os.listdir(os.path.join(pred_path, fd))
        if dataset_version == '2017':
            sub_fds = [name for name in pred_list if len(name) < 4]
            sub_fds = sorted(sub_fds)
            for sub_fd in sub_fds:
                subfd_names.append(fd+'/'+sub_fd)
        iou_seq = []
        for i,im_name in enumerate(im_list):
            iou_im = 0
            scores = []
            label_gt = np.array(Image.open(os.path.join(gt_path, fd, im_name)))
            if dataset_version == '2017':
                for j, sub_fd in enumerate(sub_fds):
                    score = np.load(os.path.join(pred_path, fd, sub_fd, im_name[:-4] + '.npy'))
                    scores.append(score)
                im_size = scores[0].shape
                bg_score = np.ones(im_size) * 0.5
                scores = [bg_score] + scores
                score_all = np.stack(tuple(scores), axis = -1)
                class_n = score_all.shape[2] - 1
                label_pred = score_all.argmax(axis=2)
            else:
                class_n = 1
                label_gt = label_gt > 0
                label_pred = np.array(Image.open(os.path.join(pred_path,fd, im_name))) > 0 
            label_pred = np.array(Image.fromarray(label_pred.astype(np.uint8)).resize((label_gt.shape[1], label_gt.shape[0]), 
                Image.NEAREST))
            #cv2.resize(label_pred, label_gt.shape, label_pred, 0, 0, cv2.INTER_NEAREST)
            if vis_path:
                im = cv2.imread(os.path.join(data_path, "JPEGImages/480p", fd, im_name.replace('png', 'jpg')))
                if class_n == 1:
                    mask_index = label_pred>0
                    im[mask_index,:] = im[mask_index,:]*0.5 + np.array(label_colours)[1]*0.5
                os.makedirs(os.path.join(vis_path, fd), exist_ok=True)
                cv2.imwrite(os.path.join(vis_path, fd, im_name), im)

            iou_seq.append(calcIoU(label_gt, label_pred, class_n))
        iou_seq = np.stack(iou_seq, axis=1)
        if logger is not None:
            logger.info ("SEQ: {}, IOU: {}".format(fd, iou_seq.mean(axis=1)))
        sample_n += iou_seq.size
        iou.extend(iou_seq.mean(axis=1).tolist())#flatten and append
    iou = np.array(iou)
    logger.info("iou: {}".format(iou.mean()))
    iou_txt_fn = "{}_iou.txt".format(pred_path)

    if dataset_version == 2017:
        with open(iou_txt_fn, "w") as f:
            for fd, num in zip(subfd_names, iou):
                f.write("%s\t%f\n" % (fd, num))
            f.write("all\t%f\n" % iou.mean())
    elif dataset_version == 2016:
        with open(iou_txt_fn, "w") as f:
            for fd, num in zip(fds, iou):
                f.write("%s\t%f\n" % (fd, num))
            f.write("all\t%f\n" % iou.mean())

        return iou.mean()

if __name__ == "__main__":
    from logger.logger import setup_logger
    logger = setup_logger()
    #data_path = sys.argv[1]
    #pred_path = sys.argv[2] #'DAVIS/Results/Segmentations/480p/OSVOS'
    #dataset_version = sys.argv[3]
    #dataset_split = sys.argv[4]
    #if len(sys.argv) > 5:
    #    vis_path = sys.argv[5]
    #else:
    #    vis_path = None
    data_path = "/data/qiang.zhou/media/DAVIS/trainval"
    #data_path = "./DAVIS/trainval"
    #pred_path = "output_epoch16_alpha1_wo_update"
    pred_path = sys.argv[1]
    dataset_version = 2016
    dataset_split = 'val'
    os.makedirs(pred_path, exist_ok=True)
    vis_path = pred_path + "_vis"
    os.makedirs(vis_path, exist_ok=True)
    compute_mIOU(data_path, pred_path, dataset_version, dataset_split, vis_path=vis_path, logger=logger)
