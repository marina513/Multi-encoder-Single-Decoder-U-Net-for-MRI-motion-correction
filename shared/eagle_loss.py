from torch import nn
import torch
import torch.nn.functional as F


class GradientVariance(nn.Module):
    """Class for calculating GV loss between to RGB images
       :parameter
       patch_size : int, scalar, size of the patches extracted from the gt and predicted images
       cpu : bool,  whether to run calculation on cpu or gpu
        """
    def __init__(self, patch_size, cpu=True):
        super(GradientVariance, self).__init__()
        self.patch_size = patch_size
        # Sobel kernel for the gradient map calculation
        self.kernel_x = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.kernel_y = torch.FloatTensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).unsqueeze(0).unsqueeze(0)
        if not cpu:
            self.kernel_x = self.kernel_x.cuda()
            self.kernel_y = self.kernel_y.cuda()
        # operation for unfolding image into non overlapping patches
        self.unfold = torch.nn.Unfold(kernel_size=(self.patch_size, self.patch_size), stride=self.patch_size)

    def forward(self, output, target):

        # calculation of the gradient maps of x and y directions
        gx_target = F.conv2d(target, self.kernel_x, stride=1, padding=1)
        gy_target = F.conv2d(target, self.kernel_y, stride=1, padding=1)
        gx_output = F.conv2d(output, self.kernel_x, stride=1, padding=1)
        gy_output = F.conv2d(output, self.kernel_y, stride=1, padding=1)

        # unfolding image to patches
        gx_target_patches = self.unfold(gx_target)
        gy_target_patches = self.unfold(gy_target)
        gx_output_patches = self.unfold(gx_output)
        gy_output_patches = self.unfold(gy_output)

        # calculation of variance of each patch
        var_target_x = torch.var(gx_target_patches, dim=1)
        var_output_x = torch.var(gx_output_patches, dim=1)
        var_target_y = torch.var(gy_target_patches, dim=1)
        var_output_y = torch.var(gy_output_patches, dim=1)

        # loss function as a MSE between variances of patches extracted from gradient maps
        gradvar_loss = F.mse_loss(var_target_x, var_output_x) + F.mse_loss(var_target_y, var_output_y)

        return gradvar_loss

























































































import torch
import torch.nn as nn
import torch.nn.functional as F


class Eagle_Loss(nn.Module):

    def __init__(self, patch_size, cpu=False, cutoff=0.5):
        super(Eagle_Loss, self).__init__()
        self.patch_size = patch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
        self.cutoff = cutoff

        # Scharr kernel for the gradient map calculation
        kernel_values = [[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]
        self.kernel_x = nn.Parameter(
            torch.tensor(kernel_values, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device),
            requires_grad=False)
        self.kernel_y = nn.Parameter(
            torch.tensor(kernel_values, dtype=torch.float32).t().unsqueeze(0).unsqueeze(0).to(self.device),
            requires_grad=False)

        # Operation for unfolding image into non-overlapping patches
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size).to(self.device)
















    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    def forward(self, output, target):

        output, target = output.to(self.device), target.to(self.device)
        if output.size(1) != 1 or target.size(1) != 1:
            raise ValueError("Input 'output' and 'target' should be either RGB (3 channels) or grayscale (1 channel)")

        # Gradient maps calculation
        gx_output, gy_output = self.calculate_gradient(output)
        gx_target, gy_target = self.calculate_gradient(target)

        # Unfolding and variance calculation
        eagle_loss = self.calculate_patch_loss(gx_output, gx_target) + \
                     self.calculate_patch_loss(gy_output, gy_target)

        return eagle_loss
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~``



    # 1) covlution with schaw kernel
    def calculate_gradient(self, img):
        img = img.to(self.device)
        gx = F.conv2d(img, self.kernel_x, padding=1, groups=img.shape[1])
        gy = F.conv2d(img, self.kernel_y, padding=1, groups=img.shape[1])
        return gx, gy







    # 2) patchize & variance
    def calculate_patch_loss(self, output_gradient, target_gradient):
        output_gradient, target_gradient = output_gradient.to(self.device), target_gradient.to(self.device)
        batch_size = output_gradient.size(0)
        num_channels = output_gradient.size(1)
        patch_size_squared = self.patch_size * self.patch_size
        output_patches = self.unfold(output_gradient).view(batch_size, num_channels, patch_size_squared, -1)
        target_patches = self.unfold(target_gradient).view(batch_size, num_channels, patch_size_squared, -1)
        
        
        
        var_output = torch.var(output_patches, dim=2, keepdim=True)
        var_target = torch.var(target_patches, dim=2, keepdim=True)

        shape0, shape1 = output_gradient.shape[-2] // self.patch_size, output_gradient.shape[-1] // self.patch_size
        return self.fft_loss(var_target.view(batch_size, num_channels, shape0, shape1), var_output.view(batch_size, num_channels, shape0, shape1))


























    def gaussian_highpass_weights2d(self, size):
        freq_x = torch.fft.fftfreq(size[0]).reshape(-1, 1).repeat(1, size[1]).to(self.device)
        freq_y = torch.fft.fftfreq(size[1]).reshape(1, -1).repeat(size[0], 1).to(self.device)

        freq_mag = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        weights = torch.exp(-0.5 * ((freq_mag - self.cutoff) ** 2))
        return 1 - weights  # Inverted for high pass

    def fft_loss(self, pred, gt):
        pred, gt = pred.to(self.device), gt.to(self.device)
        pred_fft = torch.fft.fft2(pred)
        gt_fft = torch.fft.fft2(gt)
        pred_mag = torch.sqrt(pred_fft.real ** 2 + pred_fft.imag ** 2)
        gt_mag = torch.sqrt(gt_fft.real ** 2 + gt_fft.imag ** 2)

        weights = self.gaussian_highpass_weights2d(pred.size()[2:]).to(pred.device)
        weighted_pred_mag = weights * pred_mag
        weighted_gt_mag = weights * gt_mag

        return F.l1_loss(weighted_pred_mag, weighted_gt_mag)
    