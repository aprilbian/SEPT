import torch.nn as nn
import torch
import torch.nn.functional as F
from models.utils import *
from pytorch3d.loss import chamfer_distance
import math


class get_model(nn.Module):
    def __init__(self, revise = False, normal_channel=False, bottleneck_size=256, recon_points=2048):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0

        self.revise = revise
        self.bottleneck_size = bottleneck_size
        self.normal_channel = normal_channel

        # TODO: think about how to tune the radius
        self.sa1 = PointNetSetAbstraction(npoint=512, radius=0.2, nsample=32, in_channel=3 + 3, mlp=[64, 64, 128], group_all=False)
        self.attn1 = TransformerBlock(128, 128, 32)
        self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3, mlp=[128, 128, bottleneck_size], group_all=False)
        self.attn2 = TransformerBlock(bottleneck_size, bottleneck_size, 64)

        self.attn3 = TransformerBlock(256, 256, 64)

        self.recon_points = recon_points

        self.decompression = ReconstructionLayer(recon_points // 16, bottleneck_size, 256)
        self.coor_reconstruction_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

        self.coor_upsampling1 = CoordinateUpsamplingModule(ratio=4, channels=256, radius=0.10)
        self.coor_upsampling2 = CoordinateUpsamplingModule(ratio=4, channels=256, radius=0.10)


        self.coor_reest_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )

    
        '''This is only for the power coeff'''
        self.register_buffer('x_mean', torch.tensor(0))
        self.register_buffer('x_std', torch.tensor(1))

    def forward(self, xyz, snr, global_step=None):
        B, N, C = xyz.shape

        if self.normal_channel:
            l0_feature = xyz
            l0_xyz = xyz[:, :3, :]
        else:
            l0_feature = xyz
            l0_xyz = xyz

        pc_gd = l0_xyz

        l0_xyz = l0_xyz.permute(0, 2, 1)
        l0_feature = l0_feature.permute(0, 2, 1)
        l1_xyz, l1_feature = self.sa1(l0_xyz, l0_feature)
        l1_feature, _ = self.attn1(l1_xyz.permute(0, 2, 1), l1_feature.permute(0, 2, 1))
        l2_xyz, l2_feature = self.sa2(l1_xyz, l1_feature.permute(0, 2, 1))
        x, _ = self.attn2(l2_xyz.permute(0, 2, 1), l2_feature.permute(0, 2, 1))

        # TODO: Try sth else, e.g., linear projection
        x = F.adaptive_max_pool1d(x.permute(0, 2, 1), 1).view(B, -1)          # (B, bottleneck_size)

        #x = x/torch.sqrt(torch.sum(x**2, dim = -1)/x.shape[1]).unsqueeze(-1)
        '''The 2nd pwr normalize method'''
        self.x_mean, self.x_std = torch.mean(x), torch.std(x)
        x = (x-self.x_mean)/self.x_std

        x_noise = torch.randn(x.shape).to(x.device)

        y = x + x_noise*math.sqrt(10**(-snr/10))

        # TODO: try sth other than the 1d deconv for
        #  upsampling
        # TODO: try sth to mitigate the noise
        decoder_local_feature = self.decompression(y.unsqueeze(1))

        # TODO: try to put the coordinate reconstruction later, e.g., right before the upsampling block
        new_xyz0 = self.coor_reconstruction_layer(decoder_local_feature)

        #new_feature0 = self.feature_enhence2(new_xyz0, decoder_local_feature.permute(0, 2, 1)).permute(0, 2, 1)
        new_feature0, _ = self.attn3(new_xyz0, decoder_local_feature)
        #new_feature0 = self.feature_enhence2(l2_xyz.permute(0, 2, 1), decoder_local_feature.permute(0, 2, 1)).permute(0, 2, 1)

        if self.revise:
            new_xyz0 = self.coor_reest_layer(new_feature0)

        # Any possibility to have less upsampling layer with larger K?
        new_xyz1, new_feature1 = self.coor_upsampling1(new_xyz0, new_feature0)
        #new_xyz1, new_feature1 = self.coor_upsampling1(new_xyz0, decoder_local_feature)
        #new_xyz1, new_feature1 = self.coor_upsampling1(l2_xyz.permute(0, 2, 1), new_feature0)
        new_xyz2, new_feature2 = self.coor_upsampling2(new_xyz1, new_feature1)

        
        # TODO: finetune the final coordinates using the final feature
        coor_recon = new_xyz2
        #if self.revise:
        #    refine_coor = self.coor_refine_layer(new_feature2)
        #    coor_recon = coor_recon + refine_coor

        cd = chamfer_distance(pc_gd, coor_recon)[0]

        '''Some auxillary variables to consider'''
        #noisy_cd = chamfer_distance(new_xyz, l2_xyz.permute(0,2,1))[0]
        denoised_cd = chamfer_distance(new_xyz0, l2_xyz.permute(0,2,1))[0]

        return coor_recon, cd#, denoised_cd