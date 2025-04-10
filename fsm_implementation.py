# FSM 기반 2카메라 영상 정합 및 시야 접합 시스템 구현
# Vitor Guizilini et al., "Full Surround Monodepth from Multiple Cameras", CVPR 2021

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# 1. 네트워크 구조 정의 (DepthNet, PoseNet)
# ------------------------------------------------------------

class ConvBlock(nn.Module):
    """기본 컨볼루션 블록"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBlock, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class DepthDecoder(nn.Module):
    """Depth estimation을 위한 디코더"""
    def __init__(self, num_ch_enc):
        super(DepthDecoder, self).__init__()
        
        # 디코더 레이어 정의
        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = [16, 32, 64, 128, 256]
        
        # 디코더 레이어
        self.convs = nn.ModuleDict()
        
        # 각 스케일에 대한 업샘플링 레이어 정의
        for i in range(4, -1, -1):
            # 업샘플링 입력
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            
            # 스킵 커넥션 입력을 위한 컨볼루션
            self.convs[f"upconv_{i}_0"] = ConvBlock(num_ch_in, num_ch_out)
            
            # 스킵 커넥션 + 디코더 출력 처리
            num_ch_in = num_ch_out
            if i > 0:  # 스킵 커넥션 추가
                num_ch_in += self.num_ch_enc[i - 1]
            self.convs[f"upconv_{i}_1"] = ConvBlock(num_ch_in, num_ch_out)
            
        # 최종 depth 출력 레이어
        self.depth_output = nn.Conv2d(self.num_ch_dec[0], 1, 3, padding=1)
        
    def forward(self, input_features):
        outputs = {}
        x = input_features[-1]
        
        # 각 스케일에 대한 디코딩
        for i in range(4, -1, -1):
            x = self.convs[f"upconv_{i}_0"](x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            
            if i > 0:  # 스킵 커넥션 추가
                x = torch.cat([x, input_features[i - 1]], dim=1)
                
            x = self.convs[f"upconv_{i}_1"](x)
            
            if i == 0:
                outputs["depth"] = torch.sigmoid(self.depth_output(x)) * 10.0  # 스케일 조정
            
        return outputs


class DepthNet(nn.Module):
    """ResNet18 기반 Depth Estimation 네트워크"""
    def __init__(self, pretrained=True):
        super(DepthNet, self).__init__()
        
        # ResNet18 인코더 사용
        self.encoder = models.resnet18(pretrained=pretrained)
        
        # 인코더 채널 수 정의
        num_ch_enc = [64, 64, 128, 256, 512]
        
        # 인코더의 첫 번째 레이어를 표준 RGB 입력용으로 변경
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # 디코더 초기화
        self.decoder = DepthDecoder(num_ch_enc)
        
    def forward(self, x):
        # 인코더 feature maps 추출
        features = []
        
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        features.append(self.encoder.relu(x))
        
        features.append(self.encoder.layer1(self.encoder.maxpool(features[-1])))
        features.append(self.encoder.layer2(features[-1]))
        features.append(self.encoder.layer3(features[-1]))
        features.append(self.encoder.layer4(features[-1]))
        
        # 디코더를 통해 depth 예측
        outputs = self.decoder(features)
        
        return outputs


class PoseNet(nn.Module):
    """Pose Estimation Network"""
    def __init__(self, num_input_frames=2):
        super(PoseNet, self).__init__()
        
        # 입력: 2개 이미지 (source와 target)
        self.num_input_frames = num_input_frames
        
        # 7-layer CNN 구성
        self.conv1 = ConvBlock(3 * num_input_frames, 16, 7, 2)
        self.conv2 = ConvBlock(16, 32, 5, 2)
        self.conv3 = ConvBlock(32, 64, 3, 2)
        self.conv4 = ConvBlock(64, 128, 3, 2)
        self.conv5 = ConvBlock(128, 256, 3, 2)
        self.conv6 = ConvBlock(256, 256, 3, 2)
        self.conv7 = ConvBlock(256, 256, 3, 2)
        
        # 평균 풀링 및 FC 레이어
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, 6 * (num_input_frames - 1))
        
    def forward(self, images):
        # 여러 이미지를 채널 방향으로 concatenate
        if isinstance(images, list):
            x = torch.cat(images, dim=1)
        else:
            # 이미 concatenate된 경우
            x = images
            
        # 7-layer CNN 통과
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        
        # 평균 풀링 및 FC 레이어
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        
        # 출력: (batch_size, 6)
        # [3 rotation params, 3 translation params]
        x = x.view(-1, self.num_input_frames - 1, 6)
        
        # 회전을 축-각도 표현으로 변환
        rot = x[:, :, :3]  # 회전 파라미터
        trans = x[:, :, 3:]  # 이동 파라미터
        
        return rot, trans


# ------------------------------------------------------------
# 2. Transformation 및 Warping 구현
# ------------------------------------------------------------

def euler_to_rotation_matrix(euler_angles):
    """
    오일러 각도를 회전 행렬로 변환
    euler_angles: (batch_size, 3) - [roll, pitch, yaw]
    """
    batch_size = euler_angles.shape[0]
    roll, pitch, yaw = euler_angles[:, 0], euler_angles[:, 1], euler_angles[:, 2]
    
    # Roll (X 축 기준)
    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    zeros = torch.zeros_like(cos_r)
    ones = torch.ones_like(cos_r)
    
    R_x = torch.stack([
        torch.stack([ones, zeros, zeros], dim=1),
        torch.stack([zeros, cos_r, -sin_r], dim=1),
        torch.stack([zeros, sin_r, cos_r], dim=1)
    ], dim=1)
    
    # Pitch (Y 축 기준)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    
    R_y = torch.stack([
        torch.stack([cos_p, zeros, sin_p], dim=1),
        torch.stack([zeros, ones, zeros], dim=1),
        torch.stack([-sin_p, zeros, cos_p], dim=1)
    ], dim=1)
    
    # Yaw (Z 축 기준)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
    
    R_z = torch.stack([
        torch.stack([cos_y, -sin_y, zeros], dim=1),
        torch.stack([sin_y, cos_y, zeros], dim=1),
        torch.stack([zeros, zeros, ones], dim=1)
    ], dim=1)
    
    # 최종 회전 행렬: R = R_z * R_y * R_x
    R = torch.matmul(torch.matmul(R_z, R_y), R_x)
    
    return R


def back_project(depth, intrinsics):
    """
    2D 이미지 좌표와 depth를 3D 포인트로 역투영
    depth: (B, 1, H, W)
    intrinsics: (B, 3, 3)
    """
    batch_size, _, height, width = depth.shape
    
    # 픽셀 좌표 그리드 생성
    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=depth.device),
                           torch.arange(0, width, dtype=torch.float32, device=depth.device)])
    
    x = x.contiguous().view(-1)
    y = y.contiguous().view(-1)
    
    # 균질 좌표계로 변환
    ones = torch.ones_like(x)
    pixel_coords = torch.stack([x, y, ones], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 3, H*W)
    
    # 3D 포인트 계산
    intrinsics_inv = torch.inverse(intrinsics)
    cam_points = torch.bmm(intrinsics_inv, pixel_coords)  # (B, 3, H*W)
    
    # depth 적용
    depth = depth.view(batch_size, 1, -1)
    points_3d = cam_points * depth
    
    # 동차 좌표 추가
    ones = torch.ones((batch_size, 1, points_3d.shape[2]), device=points_3d.device)
    points_3d_homogeneous = torch.cat([points_3d, ones], dim=1)  # (B, 4, H*W)
    
    return points_3d_homogeneous


def project_3d_to_2d(points_3d, intrinsics, height, width):
    """
    3D 포인트를 2D 이미지 좌표로 투영
    points_3d: (B, 4, H*W)
    intrinsics: (B, 3, 3)
    """
    batch_size = points_3d.shape[0]
    
    # 3D -> 2D 투영
    points_2d = torch.bmm(intrinsics, points_3d[:, :3, :])  # (B, 3, H*W)
    
    # 정규화
    points_2d = points_2d / (points_2d[:, 2:3, :] + 1e-7)
    
    # 이미지 내 픽셀 유효성 검사 (for masking)
    valid_x = (points_2d[:, 0, :] >= 0) & (points_2d[:, 0, :] < width)
    valid_y = (points_2d[:, 1, :] >= 0) & (points_2d[:, 1, :] < height)
    valid = valid_x & valid_y & (points_3d[:, 2, :] > 0)  # z > 0 (카메라 앞에 있음)
    
    # 결과 재구성
    points_2d = points_2d.transpose(1, 2)  # (B, H*W, 3)
    valid = valid.unsqueeze(2).float()  # (B, H*W, 1)
    
    return points_2d, valid


def warp_image(src_img, src_depth, src_intrinsics, tgt_intrinsics, rot, trans, height, width):
    """
    FSM Equation (3)에 따른 이미지 warping 구현
    src_img: 소스 이미지 (B, 3, H, W)
    src_depth: 소스 depth (B, 1, H, W)
    src_intrinsics: 소스 카메라 내부 파라미터 (B, 3, 3)
    tgt_intrinsics: 타겟 카메라 내부 파라미터 (B, 3, 3)
    rot: 회전 파라미터 (B, 3)
    trans: 이동 파라미터 (B, 3)
    """
    batch_size = src_img.shape[0]
    
    # 1. 소스 이미지 포인트를 3D로 역투영 (ϕi(pi, d^i))
    points_3d = back_project(src_depth, src_intrinsics)  # (B, 4, H*W)
    
    # 2. 회전 행렬 계산
    rotation_matrix = euler_to_rotation_matrix(rot)  # (B, 3, 3)
    
    # 3. 변환 행렬 생성
    transform_matrix = torch.zeros((batch_size, 4, 4), device=src_img.device)
    transform_matrix[:, :3, :3] = rotation_matrix
    transform_matrix[:, :3, 3] = trans
    transform_matrix[:, 3, 3] = 1.0
    
    # 4. 3D 포인트 변환 (Ri→j⋅ϕi(pi, d^i) + ti→j)
    transformed_points = torch.bmm(transform_matrix, points_3d)  # (B, 4, H*W)
    
    # 5. 변환된 3D 포인트를 타겟 카메라 좌표로 투영 (πj)
    projected_points, valid_mask = project_3d_to_2d(transformed_points, tgt_intrinsics, height, width)
    
    # 6. 그리드 샘플링을 위해 좌표 정규화 (-1, 1 범위)
    norm_points_x = 2.0 * projected_points[:, :, 0] / (width - 1) - 1.0
    norm_points_y = 2.0 * projected_points[:, :, 1] / (height - 1) - 1.0
    norm_points = torch.stack([norm_points_x, norm_points_y], dim=2)  # (B, H*W, 2)
    
    # 그리드 샘플링을 위한 형태로 변환
    norm_points = norm_points.view(batch_size, height, width, 2)
    
    # 7. 그리드 샘플링으로 워핑된 이미지 생성
    warped_img = F.grid_sample(src_img, norm_points, padding_mode='zeros', align_corners=True)
    
    # 워핑된 이미지와 유효성 마스크 반환
    valid_mask = valid_mask.view(batch_size, 1, height, width)
    
    return warped_img, valid_mask


# ------------------------------------------------------------
# 3. Loss 함수 구현
# ------------------------------------------------------------

def compute_photometric_loss(img1, img2, mask=None, alpha=0.85):
    """
    Photometric Loss 계산 (SSIM + L1)
    img1, img2: (B, 3, H, W)
    mask: (B, 1, H, W)
    """
    # SSIM 손실
    ssim_loss = 1 - compute_ssim(img1, img2)
    ssim_loss = ssim_loss.mean(1, True)
    
    # L1 손실
    l1_loss = torch.abs(img1 - img2).mean(1, True)
    
    # Combined loss
    loss = alpha * ssim_loss + (1 - alpha) * l1_loss
    
    # 마스크 적용 (필요시)
    if mask is not None:
        loss = loss * mask
        # 평균 계산 시 마스크 고려
        count = mask.sum(dim=[1, 2, 3], keepdim=True) + 1e-7
        loss = loss.sum(dim=[1, 2, 3], keepdim=True) / count
    
    return loss


def compute_ssim(img1, img2, c1=0.01**2, c2=0.03**2, window_size=11):
    """SSIM 계산"""
    # 평균 필터 생성
    mu_x = F.avg_pool2d(img1, window_size, stride=1, padding=window_size//2)
    mu_y = F.avg_pool2d(img2, window_size, stride=1, padding=window_size//2)
    
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.avg_pool2d(img1 * img1, window_size, stride=1, padding=window_size//2) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(img2 * img2, window_size, stride=1, padding=window_size//2) - mu_y_sq
    sigma_xy = F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size//2) - mu_xy

    SSIM_n = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    SSIM_d = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)
    SSIM = SSIM_n / SSIM_d
    
    return torch.clamp((1 - SSIM) / 2, 0, 1)


def compute_pose_consistency_loss(rotations, translations, alpha_t=1.0, alpha_r=1.0):
    """
    Pose Consistency Constraint Loss 계산
    rotations: 각 카메라 쌍에 대한 회전 (euler angles)
    translations: 각 카메라 쌍에 대한 이동 벡터
    """
    n_cameras = len(rotations)
    batch_size = rotations[0].shape[0]
    
    loss_trans = torch.zeros(batch_size, device=rotations[0].device)
    loss_rot = torch.zeros(batch_size, device=rotations[0].device)
    
    # 기준 카메라 (index 0)와 다른 카메라들 간의 차이 계산
    for j in range(1, n_cameras):
        # Translation loss: ||t_1 - t_j||^2
        trans_diff = translations[0] - translations[j]
        loss_trans += torch.sum(trans_diff ** 2, dim=1)
        
        # Rotation loss: sum of squared angular differences
        rot_diff_phi = rotations[0][:, 0] - rotations[j][:, 0]  # roll
        rot_diff_theta = rotations[0][:, 1] - rotations[j][:, 1]  # pitch
        rot_diff_psi = rotations[0][:, 2] - rotations[j][:, 2]  # yaw
        
        loss_rot += rot_diff_phi ** 2 + rot_diff_theta ** 2 + rot_diff_psi ** 2
    
    # 최종 pose consistency loss
    loss_pcc = alpha_t * loss_trans + alpha_r * loss_rot
    
    return loss_pcc


# ------------------------------------------------------------
# 4. 전체 시스템 통합 및 추론 파이프라인
# ------------------------------------------------------------

class FSMDualCameraSystem:
    """FSM 기반 2카메라 영상 정합 및 시야 접합 시스템"""
    def __init__(self, img_height=256, img_width=512):
        self.img_height = img_height
        self.img_width = img_width
        
        # 네트워크 초기화
        self.depth_net = DepthNet(pretrained=True)
        self.pose_net = PoseNet(num_input_frames=2)
        
        # 디바이스 설정
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth_net.to(self.device)
        self.pose_net.to(self.device)
        
        self.alpha_t = 1.0  # Translation loss 가중치
        self.alpha_r = 1.0  # Rotation loss 가중치
        
    def preprocess_image(self, img):
        """이미지 전처리"""
        # OpenCV로 읽은 이미지는 BGR -> RGB로 변환
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 크기 조정
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # 정규화 [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # 차원 변환 (H, W, C) -> (C, H, W)
        img = np.transpose(img, (2, 0, 1))
        
        # 텐서 변환
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        return img
    
    def predict_depth(self, img):
        """단일 이미지의 depth map 예측"""
        with torch.no_grad():
            outputs = self.depth_net(img)
            depth = outputs["depth"]
        return depth
    
    def predict_pose(self, img1, img2):
        """두 이미지 간의 상대적 pose 예측"""
        with torch.no_grad():
            # 두 이미지를 채널 방향으로 연결
            stacked_imgs = torch.cat([img1, img2], dim=1)
            rot, trans = self.pose_net(stacked_imgs)
            rot = rot.squeeze(1)  # (B, 3)
            trans = trans.squeeze(1)  # (B, 3)
        return rot, trans
    
    def process_dual_camera_images(self, img1, img2, K1, K2):
        """
        두 카메라 이미지 처리 및 통합
        img1, img2: 카메라 이미지 (numpy arrays)
        K1, K2: 카메라 내부 파라미터 (numpy arrays)
        """
        # 1. 이미지 전처리
        img1_tensor = self.preprocess_image(img1)
        img2_tensor = self.preprocess_image(img2)
        
        # 2. Depth 예측
        depth1 = self.predict_depth(img1_tensor)
        depth2 = self.predict_depth(img2_tensor)
        
        # 3. 카메라 간 Pose 예측
        rot12, trans12 = self.predict_pose(img1_tensor, img2_tensor)
        
        # 4. 내부 파라미터 텐서 변환
        K1_tensor = torch.from_numpy(K1.astype(np.float32)).unsqueeze(0).to(self.device)
        K2_tensor = torch.from_numpy(K2.astype(np.float32)).unsqueeze(0).to(self.device)
        
        # 5. 이미지 Warping (카메라 1 -> 카메라 2)
        warped_img1_to_2, mask1_to_2 = warp_image(
            img1_tensor, depth1, K1_tensor, K2_tensor, 
            rot12, trans12, self.img_height, self.img_width
        )
        
        # 6. 역방향 Pose 계산 (카메라 2 -> 카메라 1)
        # 단순 변환: -rotation, -translation (완벽하진 않지만 근사치)
        rot21 = -rot12
        trans21 = -trans12
        
        # 7. 이미지 Warping (카메라 2 -> 카메라 1)
        warped_img2_to_1, mask2_to_1 = warp_image(
            img2_tensor, depth2, K2_tensor, K1_tensor, 
            rot21, trans21, self.img_height, self.img_width
        )
        
        # 8. 통합된 시야 생성
        # 마스크 기반으로 이미지 블렌딩
        combined_view = self.blend_images(img1_tensor, warped_img2_to_1, mask2_to_1)
        
        # 9. 결과 후처리
        depth1_np = depth1.squeeze().cpu().numpy()
        depth2_np = depth2.squeeze().cpu().numpy()
        warped_img1_to_2_np = warped_img1_to_2.squeeze().permute(1, 2, 0).cpu().numpy()
        warped_img2_to_1_np = warped_img2_to_1.squeeze().permute(1, 2, 0).cpu().numpy()
        combined_view_np = combined_view.squeeze().permute(1, 2, 0).cpu().numpy()
        
        # 시각화를 위해 값 범위 조정
        combined_view_np = np.clip(combined_view_np, 0, 1)
        
        return {
            'depth1': depth1_np,
            'depth2': depth2_np,
            'warped_img1_to_2': warped_img1_to_2_np,
            'warped_img2_to_1': warped_img2_to_1_np,
            'combined_view': combined_view_np,
            'pose_rotation': rot12.cpu().numpy(),
            'pose_translation': trans12.cpu().numpy()
        }
    
    def blend_images(self, img1, img2_warped, mask2):
        """두 이미지를 마스크 기반으로 블렌딩"""
        # 겹치는 영역에 대한 블렌딩
        overlap_weight = F.avg_pool2d(mask2, 31, stride=1, padding=15)
        
        # 첫 번째 이미지는 그대로 유지하고 두 번째 이미지는 워핑된 영역만 사용
        blended = img1 * (1 - overlap_weight) + img2_warped * overlap_weight
        
        return blended
def train_step(self, img1, img2, K1, K2):
        """
        단일 학습 스텝 실행
        img1, img2: 카메라 이미지 (B, 3, H, W)
        K1, K2: 카메라 내부 파라미터 (B, 3, 3)
        """
        # 그라디언트 초기화
        self.depth_net.train()
        self.pose_net.train()
        
        # 1. Depth 예측
        outputs1 = self.depth_net(img1)
        outputs2 = self.depth_net(img2)
        depth1 = outputs1["depth"]
        depth2 = outputs2["depth"]
        
        # 2. Pose 예측
        stacked_1_2 = torch.cat([img1, img2], dim=1)
        rot12, trans12 = self.pose_net(stacked_1_2)
        rot12 = rot12.squeeze(1)
        trans12 = trans12.squeeze(1)
        
        # 반대 방향 pose (근사치)
        rot21 = -rot12
        trans21 = -trans12
        
        # 3. 이미지 warping
        warped_img1_to_2, mask1_to_2 = warp_image(
            img1, depth1, K1, K2, rot12, trans12, 
            self.img_height, self.img_width
        )
        
        warped_img2_to_1, mask2_to_1 = warp_image(
            img2, depth2, K2, K1, rot21, trans21, 
            self.img_height, self.img_width
        )
        
        # 4. 마스크 계산 - 겹침 영역과 자체 가림 마스크
        # Non-overlapping mask (warping 유효성 마스크)
        M_no_1to2 = mask1_to_2
        M_no_2to1 = mask2_to_1
        
        # Self-occlusion mask
        # depth 차이가 큰 경우 자체 가림 영역으로 판단
        with torch.no_grad():
            # 원본 depth와 warping된 이미지의 depth 비교
            depth2_warped = F.grid_sample(
                depth2, mask1_to_2.permute(0, 2, 3, 1), 
                padding_mode='zeros', align_corners=True
            )
            
            depth1_warped = F.grid_sample(
                depth1, mask2_to_1.permute(0, 2, 3, 1), 
                padding_mode='zeros', align_corners=True
            )
            
            # 자체 가림 영역 마스크 (depth 차이가 큰 영역)
            occlusion_threshold = 0.1
            M_so_1to2 = (torch.abs(depth1 - depth2_warped) < occlusion_threshold).float()
            M_so_2to1 = (torch.abs(depth2 - depth1_warped) < occlusion_threshold).float()
        
        # 5. 최종 마스크 계산 (겹침 영역 + 자체 가림 마스크)
        final_mask_1to2 = M_no_1to2 * M_so_1to2
        final_mask_2to1 = M_no_2to1 * M_so_2to1
        
        # 6. Photometric Loss 계산
        loss_photo_1to2 = compute_photometric_loss(img2, warped_img1_to_2, final_mask_1to2)
        loss_photo_2to1 = compute_photometric_loss(img1, warped_img2_to_1, final_mask_2to1)
        
        # 전체 photometric loss
        loss_photo = (loss_photo_1to2 + loss_photo_2to1).mean()
        
        # 7. Pose Consistency Loss 계산
        # FSM에서는 여러 카메라를 고려하지만, 우리는 2대만 사용하므로 단순화
        # 두 방향의 pose가 일관성을 가져야 함 (rot12 ≈ -rot21, trans12 ≈ -trans21)
        loss_pose = compute_pose_consistency_loss(
            [rot12, rot21], [trans12, trans21], 
            self.alpha_t, self.alpha_r
        ).mean()
        
        # 8. 전체 Loss
        loss = loss_photo + 0.1 * loss_pose
        
        return loss, {
            'loss_photo': loss_photo.item(),
            'loss_pose': loss_pose.item(),
            'total_loss': loss.item()
        }
    
def train(self, train_loader, optimizer, num_epochs=10):
    """
    모델 학습
    train_loader: 학습 데이터 로더
    optimizer: 옵티마이저
    """
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for batch_idx, (img1, img2, K1, K2) in enumerate(train_loader):
            img1 = img1.to(self.device)
            img2 = img2.to(self.device)
            K1 = K1.to(self.device)
            K2 = K2.to(self.device)
            
            # 학습 단계
            optimizer.zero_grad()
            loss, loss_dict = self.train_step(img1, img2, K1, K2)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, '
                        f'Loss: {loss.item():.4f}, Photo: {loss_dict["loss_photo"]:.4f}, '
                        f'Pose: {loss_dict["loss_pose"]:.4f}')
        
        avg_loss = epoch_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{num_epochs} completed, Avg Loss: {avg_loss:.4f}')

def save_models(self, depth_path='depth_net.pth', pose_path='pose_net.pth'):
    """모델 저장"""
    torch.save(self.depth_net.state_dict(), depth_path)
    torch.save(self.pose_net.state_dict(), pose_path)
    print(f"Models saved to {depth_path} and {pose_path}")

def load_models(self, depth_path='depth_net.pth', pose_path='pose_net.pth'):
    """모델 로드"""
    self.depth_net.load_state_dict(torch.load(depth_path, map_location=self.device))
    self.pose_net.load_state_dict(torch.load(pose_path, map_location=self.device))
    print(f"Models loaded from {depth_path} and {pose_path}")


# ------------------------------------------------------------
# 5. 시각화 유틸리티
# ------------------------------------------------------------

def visualize_results(results):
    """결과 시각화"""
    # 결과 시각화를 위한 서브플롯 생성
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    # 원본 이미지 표시
    if 'img1' in results:
        axes[0, 0].imshow(results['img1'])
        axes[0, 0].set_title('Camera 1 Image')
    
    if 'img2' in results:
        axes[0, 1].imshow(results['img2'])
        axes[0, 1].set_title('Camera 2 Image')
    
    # Depth 맵 표시
    if 'depth1' in results:
        axes[1, 0].imshow(results['depth1'], cmap='magma')
        axes[1, 0].set_title('Depth Map 1')
    
    if 'depth2' in results:
        axes[1, 1].imshow(results['depth2'], cmap='magma')
        axes[1, 1].set_title('Depth Map 2')
    
    # Warped 이미지 표시
    if 'warped_img2_to_1' in results:
        axes[0, 2].imshow(results['warped_img2_to_1'])
        axes[0, 2].set_title('Camera 2 → Camera 1 Warping')
    
    # 통합 뷰 표시
    if 'combined_view' in results:
        axes[1, 2].imshow(results['combined_view'])
        axes[1, 2].set_title('Combined Panorama View')
    
    # 레이아웃 조정
    plt.tight_layout()
    plt.show()


def create_panorama(img1, img2, fsm_system):
    """두 카메라 이미지로부터 파노라마 생성"""
    # 이미지 전처리
    img1_tensor = fsm_system.preprocess_image(img1)
    img2_tensor = fsm_system.preprocess_image(img2)
    
    # Depth 예측
    depth1 = fsm_system.predict_depth(img1_tensor)
    depth2 = fsm_system.predict_depth(img2_tensor)
    
    # 두 카메라 간의 Pose 예측
    rot12, trans12 = fsm_system.predict_pose(img1_tensor, img2_tensor)
    
    # 내부 파라미터 (예시 값으로 시작, 실제 사용 시 정확한 값으로 대체)
    # 70도 FOV를 가진 카메라의 내부 파라미터 계산
    fx = fsm_system.img_width / (2 * np.tan(np.radians(70) / 2))
    fy = fx  # 정사각형 픽셀 가정
    cx = fsm_system.img_width / 2
    cy = fsm_system.img_height / 2
    
    K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    K2 = K1.copy()  # 두 카메라가 동일한 내부 파라미터를 가진다고 가정
    
    K1_tensor = torch.from_numpy(K1).unsqueeze(0).to(fsm_system.device)
    K2_tensor = torch.from_numpy(K2).unsqueeze(0).to(fsm_system.device)
    
    # 두 이미지 Warping
    with torch.no_grad():
        # 카메라 1 -> 카메라 2
        warped_img1_to_2, mask1_to_2 = warp_image(
            img1_tensor, depth1, K1_tensor, K2_tensor, 
            rot12, trans12, fsm_system.img_height, fsm_system.img_width
        )
        
        # 카메라 2 -> 카메라 1 (반대 방향)
        rot21 = -rot12  # 근사치
        trans21 = -trans12  # 근사치
        
        warped_img2_to_1, mask2_to_1 = warp_image(
            img2_tensor, depth2, K2_tensor, K1_tensor, 
            rot21, trans21, fsm_system.img_height, fsm_system.img_width
        )
    
    # 파노라마 생성 (블렌딩)
    panorama = fsm_system.blend_images(img1_tensor, warped_img2_to_1, mask2_to_1)
    
    # 결과 후처리
    panorama_np = panorama.squeeze().permute(1, 2, 0).cpu().numpy()
    panorama_np = np.clip(panorama_np, 0, 1)
    
    return panorama_np


# ------------------------------------------------------------
# 6. 데이터 로더 (학습 시 필요)
# ------------------------------------------------------------

class DualCameraDataset(torch.utils.data.Dataset):
    """
    두 카메라 영상 데이터셋 
    - 실제 구현 시 데이터 경로 및 로딩 로직은 수정 필요
    """
    def __init__(self, data_dir, img_height=256, img_width=512, transform=None):
        self.data_dir = data_dir
        self.img_height = img_height
        self.img_width = img_width
        self.transform = transform
        
        # 데이터 파일 목록 로드
        # 예시 구현이므로 실제 데이터셋에 맞게 수정 필요
        self.samples = []
        
        # TODO: 실제 데이터셋 경로에서 이미지 쌍 목록 로드
        # self.samples = [파일 경로 리스트]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # 샘플 로드
        sample_path = self.samples[idx]
        
        # TODO: 실제 데이터셋에 맞게 이미지와 카메라 내부 파라미터 로드 로직 구현
        # 예시:
        # img1 = cv2.imread(sample_path[0])
        # img2 = cv2.imread(sample_path[1])
        # K1 = np.load(sample_path[2])
        # K2 = np.load(sample_path[3])
        
        # 테스트용 더미 데이터
        img1 = np.random.rand(self.img_height, self.img_width, 3).astype(np.float32)
        img2 = np.random.rand(self.img_height, self.img_width, 3).astype(np.float32)
        
        # 예시 내부 파라미터 (70도 FOV)
        fx = self.img_width / (2 * np.tan(np.radians(70) / 2))
        fy = fx
        cx = self.img_width / 2
        cy = self.img_height / 2
        
        K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        K2 = K1.copy()  # 두 카메라가 동일하다고 가정
        
        # 이미지 변환
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        else:
            # 기본 변환 (numpy -> tensor)
            img1 = torch.from_numpy(np.transpose(img1, (2, 0, 1)))
            img2 = torch.from_numpy(np.transpose(img2, (2, 0, 1)))
        
        # 내부 파라미터 텐서 변환
        K1 = torch.from_numpy(K1)
        K2 = torch.from_numpy(K2)
        
        return img1, img2, K1, K2


# ------------------------------------------------------------
# 7. 실행 예제
# ------------------------------------------------------------

def main():
    """시스템 실행 예제"""
    # FSM 시스템 초기화
    fsm_system = FSMDualCameraSystem(img_height=256, img_width=512)
    
    # 두 카메라 이미지 로드 (예시)
    # 실제 사용 시 카메라 영상 또는 이미지 파일로 대체
    img1 = np.random.rand(512, 512, 3).astype(np.float32)
    img2 = np.random.rand(512, 512, 3).astype(np.float32)
    
    # 카메라 내부 파라미터 설정 (예시)
    # 70도 FOV를 가진 카메라의 내부 파라미터 계산
    fx = 512 / (2 * np.tan(np.radians(70) / 2))
    fy = fx
    cx = 512 / 2
    cy = 512 / 2
    
    K1 = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    K2 = K1.copy()  # 두 카메라가 동일한 내부 파라미터를 가진다고 가정
    
    # 시스템으로 이미지 처리
    results = fsm_system.process_dual_camera_images(img1, img2, K1, K2)
    
    # 결과에 원본 이미지 추가
    results['img1'] = img1
    results['img2'] = img2
    
    # 결과 시각화
    visualize_results(results)
    
    # 파노라마 생성
    panorama = create_panorama(img1, img2, fsm_system)
    
    # 파노라마 표시
    plt.figure(figsize=(12, 6))
    plt.imshow(panorama)
    plt.title('Panorama from Dual Cameras')
    plt.axis('off')
    plt.show()
    
    print("FSM Dual Camera System 처리 완료")


if __name__ == "__main__":
    main()    