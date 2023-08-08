import collections

import open3d
import os
import numpy as np
import torch
from dataset import Dataset
from torch.utils.data import DataLoader
from utils.pc_error_wrapper import pc_error
import time
import importlib
import sys
import argparse
import logging
from logging import handlers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
device = torch.device('cuda:0')


class Logger(object):
    level_relations = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'crit': logging.CRITICAL
    }

    def __init__(self, filename, level='info', when='D', backCount=3,
                 fmt='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'):
        self.logger = logging.getLogger(filename)
        format_str = logging.Formatter(fmt)
        self.logger.setLevel(self.level_relations.get(level))
        sh = logging.StreamHandler()
        sh.setFormatter(format_str)
        th = handlers.TimedRotatingFileHandler(filename=filename, when=when, backupCount=backCount, encoding='utf-8')
        th.setFormatter(format_str)
        self.logger.addHandler(sh)
        self.logger.addHandler(th)


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('point_based_PCGC')
    parser.add_argument('--dataset_path', type=str, default='./shapenetv2')
    parser.add_argument('--bottleneck_size', default=300, type=int)
    parser.add_argument('--recon_points', default=2048, type=int)
    parser.add_argument('--snr', default=10, help='SNR of the AWGN channel')
    return parser.parse_args()


def cal_d2(pc_gt, decoder_output, step, checkpoint_path):

    ori_pcd = open3d.geometry.PointCloud()
    ori_pcd.points = open3d.utility.Vector3dVector(np.squeeze(pc_gt))
    ori_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)) 
    orifile = checkpoint_path + '/pc_file/' + 'd2_ori_' + str(step) + '.ply'
    print(orifile)
    open3d.io.write_point_cloud(orifile, ori_pcd, write_ascii=True)

    lines = open(orifile).readlines()
    to_be_modified = [7, 8, 9]
    for i in to_be_modified:
        lines[i] = lines[i].replace('double', 'float32')
    file = open(orifile, 'w')
    for line in lines:
        file.write(line)
    file.close()



    rec_pcd = open3d.geometry.PointCloud()
    rec_pcd.points = open3d.utility.Vector3dVector(np.squeeze(decoder_output))

    recfile = checkpoint_path + '/pc_file/' + 'd2_rec_' + str(step) + '.ply'
    open3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)

    pc_error_metrics = pc_error(infile1=orifile, infile2=recfile, normal=True, res=2)
    pc_errors = [pc_error_metrics["mseF      (p2point)"][0],
                 pc_error_metrics["mseF,PSNR (p2point)"][0],
                 pc_error_metrics["mse1,PSNR (p2plane)"][0],
                 pc_error_metrics["mse2,PSNR (p2plane)"][0],
                 pc_error_metrics["mseF,PSNR (p2plane)"][0],
                 pc_error_metrics["mse1      (p2plane)"][0],
                 pc_error_metrics["mse1      (p2plane)"][0],
                 pc_error_metrics["mse1      (p2plane)"][0],
                 pc_error_metrics["mse2      (p2plane)"][0],
                 pc_error_metrics["mseF      (p2plane)"][0]]

    return pc_errors


def test(model, args, batch_size=1):
    
    model_path = args.exp_dir 
    checkpoint_path = model_path + '/checkpoints'

    test_data = Dataset(root=args.dataset_path, dataset_name='shapenetcorev2', num_points=args.recon_points, split='test')

    test_loader = DataLoader(test_data, num_workers=2, batch_size=batch_size, shuffle=False)

    avg_chamfer_dist = np.array([0.0 for i in range(55)])
    avg_d1_psnr = np.array([0.0 for i in range(55)])
    avg_d1_mse = np.array([0.0 for i in range(55)])
    avg_d2_psnr = np.array([0.0 for i in range(55)])
    avg_d2_mse = np.array([0.0 for i in range(55)])
    counter = np.array([0.0 for i in range(55)])
    total_chamfer_dist = 0.0
    total_noisy_cd = 0.0
    total_denoise_cd = 0.0
    total_d1_psnr = 0.0
    total_d1_mse = 0.0
    total_d2_psnr = 0.0
    total_d2_mse = 0.0

    num_samples = 0

    if not os.path.exists(model_path + '/logs'):
        os.makedirs(model_path + '/logs')
    if not os.path.exists(checkpoint_path + '/pc_file'):
        os.makedirs(checkpoint_path + '/pc_file')
    log = Logger(model_path + '/logs/result.txt', level='debug')

    for step, data in enumerate(test_loader):
        with torch.no_grad():
            pc_data = data[0]
            label = data[1]

            if torch.cuda.is_available():
                pc_gt = pc_data.to(device)
                pc_data = pc_data.to(device)

            decoder_output, cd = model(pc_data, snr = args.snr)
            #decoder_output, cd = model(pc_data, snr = args.snr)
            #decoder_output, cd, noisy_cd, denoise_cd = model(pc_data, snr = args.snr)


            # convert to numpy
            pc_gt = pc_gt.cpu().detach().numpy()
            decoder_output = decoder_output.cpu().detach().numpy()
            cd = cd.cpu().detach().numpy()
            #noisy_cd = noisy_cd.cpu().detach().numpy()
            #denoise_cd = denoise_cd.cpu().detach().numpy()

            # D1, D2 psnr & mse
            d2_results = cal_d2(pc_gt, decoder_output, step, checkpoint_path)
            d1_psnr = d2_results[1].item()
            d1_mse = d2_results[0].item()
            avg_d1_mse[label] += d1_mse
            total_d1_mse += d1_mse
            avg_d1_psnr[label] += d1_psnr
            total_d1_psnr += d1_psnr
            
            d2_psnr = d2_results[4].item()
            d2_mse = d2_results[7].item()
            avg_d2_mse[label] += d2_mse
            total_d2_mse += d2_mse
            avg_d2_psnr[label] += d2_psnr
            total_d2_psnr += d2_psnr

            # Chamfer distance
            total_chamfer_dist += cd
            #total_noisy_cd += noisy_cd
            #total_denoise_cd += denoise_cd
            avg_chamfer_dist[label] += cd

            log.logger.info(f"step: {step}")
            log.logger.info(f"d1_psnr: {d1_psnr}")
            log.logger.info(f"d2_psnr: {d2_psnr}")
            log.logger.info(f"chamfer_dist: {cd}")
            #log.logger.info(f"chamfer_dist: {noisy_cd}")
            #log.logger.info(f"chamfer_dist: {denoise_cd}")
            log.logger.info("-----------------------------------------------------")

        counter[label] += 1
        num_samples += 1
    for i in range(55):
        avg_chamfer_dist[i] /= counter[i]
        avg_d1_psnr[i] /= counter[i]
        avg_d1_mse[i] /= counter[i]
        avg_d2_psnr[i] /= counter[i]
        avg_d2_mse[i] /= counter[i]
    
    total_chamfer_dist /= num_samples
    total_denoise_cd /= num_samples
    total_noisy_cd /= num_samples
    total_d1_psnr /= num_samples
    total_d2_mse /= num_samples
    total_d2_psnr /= num_samples
    total_d2_mse /= num_samples

    for i in range(55):
        outstr = str(
            i) + " Average_D1_PSNR: %.6f, Average_D1_mse: %.6f, Average_Chamfer_Dist: %.6f, Average_D2_PSNR: %.6f, Average_D2_mse: %.6f\n" % (
                     avg_d1_psnr[i], avg_d1_mse[i], avg_chamfer_dist[i], avg_d2_psnr[i], avg_d2_mse[i])
        print(outstr)
        log.logger.info(f"{outstr}")

    outstr = "Average_D1_PSNR: %.6f, Average_D1_mse: %.6f, Average_Chamfer_Dist: %.6f, Average_Noisy_cd: %.6f, Average_Denoise_cd: %.6f, Average_D2_PSNR: %.6f, Average_D2_mse: %.6f\n" % (
        total_d1_psnr, total_d1_mse, total_chamfer_dist, total_noisy_cd, total_denoise_cd, total_d2_psnr, total_d2_mse)
    log.logger.info(f"{outstr}")
    print(outstr)


if __name__ == '__main__':

    print(torch.cuda.device_count())
    print(torch.cuda.is_available())

    args = parse_args()
    

    model_name = 'PCT_trans_ptv1'
    exp_name = str(args.bottleneck_size) + '_' + str(args.recon_points) + '_snr' + str(args.snr)

    experiment_dir = 'log/' + model_name + '/' + exp_name
    args.exp_dir = experiment_dir

    MODEL = importlib.import_module(model_name)
    print(MODEL)
    model = MODEL.get_model(revise = True, bottleneck_size = args.bottleneck_size, recon_points = args.recon_points).to(device)
    model.eval()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    new_state_dict = collections.OrderedDict()
    for k, v in checkpoint['model_state_dict'].items():
        name = k.replace('module.', '')
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)

    start_time = time.time()
    test(model, args)
    end_time = time.time()
    outstr = "test_time: %.6f" % ((end_time - start_time))
    print(outstr)
