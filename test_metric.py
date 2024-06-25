import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F

from lib.Network import Network

from utils.data_val import test_dataset
import utils.metrics as Measure


def test_model(test_loader, model):
    WFM = Measure.WeightedFmeasure()
    SM = Measure.Smeasure()
    EM = Measure.Emeasure()
    MAE = Measure.MAE()

    model.eval()
    with torch.no_grad():
        with tqdm(total=len(test_loader)) as pbar:
            for i in range(test_loader.size):
                # image, gt, edge, name, image_for_post = test_loader.load_data()
                image, gt, name, _ = test_loader.load_data()
                image = image.cuda()
                # gt = gt.numpy().astype(np.float32).squeeze()
                gt = np.asarray(gt).astype(np.float32).squeeze()
                gt /= (gt.max() + 1e-8)

                _, _, _, _, res, _, _, _, _ = model(image)

                res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)  # 标准化处理,把数值范围控制到(0,1)

                WFM.step(pred=res * 255, gt=gt * 255)
                SM.step(pred=res * 255, gt=gt * 255)
                EM.step(pred=res * 255, gt=gt * 255)
                MAE.step(pred=res * 255, gt=gt * 255)

                pbar.update()

        sm = SM.get_results()['sm'].round(3)
        adpem = EM.get_results()['em']['adp'].round(3)
        wfm = WFM.get_results()['wfm'].round(3)
        mae = MAE.get_results()['mae'].round(3)

        return {'M': mae, 'Sm': sm, 'adpE': adpem, 'wF': wfm}



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='r2cnet')
    parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
    parser.add_argument('--shot', type=int, default=5)

    parser.add_argument('--num_workers', type=int, default=12, help='the number of workers in dataloader')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')

    parser.add_argument('--data_root', type=str, default='/home/ljh/ljh_COD/DRNet/Test/',
                        help='the path to put dataset')
    parser.add_argument('--save_root', type=str, default='', help='the path to save model params and log')
    parser.add_argument('--model_epoch', type=str, default='Net_epoch_best', help='the path to train model')


    opt = parser.parse_args()
    print(opt)

    # load model
    model = Network(64).cuda()
    params_path = os.path.join(opt.save_root, '{}.pth'.format(opt.model_epoch))
    model.load_state_dict(
        torch.load(params_path))

    print('已加载{}的模型参数'.format(params_path))
    model.cuda()

    # load data
    dataset_path = opt.data_root
    test_datasets = ['CAMO', 'COD10K', 'NC4K']
    # num = 2

    for num in range(3):
        image_root = dataset_path + test_datasets[num] + '/Imgs/'
        gt_root = dataset_path + test_datasets[num] + '/GT/'
        # edge_root = dataset_path + test_datasets[num] + '/Edge/'
        test_loader = test_dataset(image_root, gt_root, 384)

        # processing
        scores = test_model(test_loader, model)

        print('{}的分数为'.format(test_datasets[num]), scores)
        print('*' * 50)

