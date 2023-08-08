import argparse
import numpy as np
import os
import torch
import logging
from pathlib import Path
import sys
import importlib
import shutil
from dataset import Dataset
from torch import nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('PCGC_PCT')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training [default: 32]')
    parser.add_argument('--model', default='PCT_trans_ptv1', help='model name')
    parser.add_argument('--revise', default=True, help='coordinate re-estimation module')
    parser.add_argument('--snr', default=10, help='SNR of the AWGN channel')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training [default: 100]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='1', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate [default: 1e-4]')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--bottleneck_size', default=300, type=int)
    parser.add_argument('--recon_points', default=2048, type=int)
    parser.add_argument('--pretrained', default='', type=str)
    parser.add_argument('--dataset_path', type=str, default='./shapenetv2')
    return parser.parse_args()


def test(args, model, loader, global_epoch=None):
    mean_loss = []
    mean_cd = []
    length = len(loader)
    with torch.no_grad():
        for j, data in enumerate(loader):
            if j % 100 == 0:
                print(j, '/', length)
            points = data[0]
            points = points.cuda()
            model.eval()
            _, cd = model(points, snr = args.snr)
            loss = cd #+ 0.1* aux_cd
            mean_cd.append(cd.mean().item())
            mean_loss.append(loss.mean().item())
    return np.mean(mean_loss), np.mean(mean_cd)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    experiment_name = str(args.bottleneck_size) + '_' + str(args.recon_points) + '_snr' + str(args.snr)
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath(args.model)
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(experiment_name)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)

    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    TRAIN_DATASET = Dataset(root=args.dataset_path, dataset_name='shapenetcorev2', num_points=2048, split='train')
    VAL_DATASET = Dataset(root=args.dataset_path, dataset_name='shapenetcorev2', num_points=2048, split='val')
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=4)
    valDataLoader = torch.utils.data.DataLoader(VAL_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=4)

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/utils.py', str(experiment_dir))


    model = MODEL.get_model(revise = args.revise, bottleneck_size=args.bottleneck_size, recon_points=args.recon_points).cuda()

    '''pretrain or train from scratch'''
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    '''finetune'''
    try:
        checkpoint = torch.load(args.pretrained)
        start_epoch = 0
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Finetuning')
    except:
        log_string('No pretrained model')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate)


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    global_step = 0
    if args.log_dir:
        best_loss_test = checkpoint['loss']
    else:
        best_loss_test = 1e9

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        mean_loss = []
        mean_cd_loss = []
        log_string('Epoch %d (%d/%s):' % (epoch + 1, epoch + 1, args.epoch))
        scheduler.step()
        length = len(trainDataLoader)
        for batch_id, data in enumerate(trainDataLoader, 0):
            if batch_id % 1000 == 0:
                print(batch_id, '/', length)
            points = data[0]

            if not args.cpu:
                points = points.cuda()
            else:
                points = points
            optimizer.zero_grad()

            model.train()

            _, cd = model(points, snr = args.snr, global_step=global_step)
            loss = 1000*cd           # 1000 is a hyper-parameter, does not change the final results

            if args.multigpu:
                loss = loss.mean()
                cd = cd.mean()

            loss.backward()
            optimizer.step()
            mean_loss.append(loss.item())
            mean_cd_loss.append(cd.item())
            global_step += 1
        ml = np.mean(mean_loss)
        mcd = np.mean(mean_cd_loss)
        log_string('mean loss: %f' % ml)
        log_string('mean chamfer distance: %f' % mcd)

        if epoch % 2 == 0:
            log_string('Start val...')
            with torch.no_grad():
                mean_loss_test, mean_cd_test = test(args, model.eval(), valDataLoader, global_step)
                log_string('val loss: %f' % (mean_loss_test))
                log_string('val cd: %f' % (mean_cd_test))

                writer.add_scalars('Validate/chamfer_dist',
                           {f'{str(experiment_name)}': mean_cd_test}, epoch)

                if (mean_loss_test < best_loss_test and epoch >= 30):
                    logger.info('Save model...')
                    best_loss_test = mean_loss_test
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s' % savepath)
                    state = {
                        'epoch': epoch,
                        'loss': mean_loss_test,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
