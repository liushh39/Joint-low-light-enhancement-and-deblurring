import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.archs.lednet_submodules import *
from basicsr.utils.registry import ARCH_REGISTRY
import functools
import basicsr.archs.arch_util as arch_util


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention4(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # print(q.shape)
        q = self.layer_norm(q)
        k = self.layer_norm(k)
        v = self.layer_norm(v)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # print(attn.shape, '2')
        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        # q = self.layer_norm(q)
        return q, attn


class PositionwiseFeedForward4(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x
        x = self.layer_norm(x)
        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        return x

class EncoderLayer3(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer3, self).__init__()
        self.slf_attn = MultiHeadAttention4(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward4(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, enc_slf_attn

class Encoder_patch66(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, d_word_vec=516, n_layers=6, n_head=8, d_k=64, d_v=64,
                 d_model=576, d_inner=2048, dropout=0.0, n_position=10, scale_emb=False):
        # 2048
        super().__init__()

        self.n_position = n_position
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer3(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.count = 0
        self.center_example = None
        self.center_coordinate = None

    def forward(self, src_fea, src_location, return_attns=False, src_mask=None):
        enc_output = src_fea
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(enc_output, slf_attn_mask=src_mask)
        return enc_output



class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.PReLU()
            ))
        self.features = nn.ModuleList(self.features)
        self.fuse = nn.Sequential(
                nn.Conv2d(in_dim+reduction_dim*4, in_dim, kernel_size=3, padding=1, bias=False),
                nn.PReLU())

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        out_feat = self.fuse(torch.cat(out, 1))
        return out_feat


class MCP(nn.Module):
    def __init__(self, nf=64):
        super(MCP, self).__init__()

        self.conv_dila1 = nn.Conv2d(nf, nf, kernel_size=3, dilation=4, padding=4, )
        self.conv_dila2 = nn.Conv2d(nf, nf, kernel_size=3, dilation=3, padding=3, )
        self.conv_dila3 = nn.Conv2d(nf, nf, kernel_size=3, dilation=2, padding=2, )
        self.conv_dila4 = nn.Conv2d(nf, nf, kernel_size=3, dilation=1, padding=1, )

        self.conv_first1 = nn.Conv2d(nf * 4, nf, 1, 1, 0, bias=True)
        self.depth_conv1 = nn.Conv2d(nf, nf, kernel_size=3, padding=1, groups=nf)
        self.depth_conv2 = nn.Conv2d(nf, nf, kernel_size=3, padding=1, groups=nf)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_fea):

        # B1
        images_lv1 = in_fea
        H = images_lv1.size(2)
        W = images_lv1.size(3)
        # B2
        images_lv2_1 = images_lv1[:, :, 0:int(H / 2), :]
        images_lv2_2 = images_lv1[:, :, int(H / 2):H, :]
        # B3
        images_lv3_1 = images_lv2_1[:, :, :, 0:int(W / 2)]
        images_lv3_2 = images_lv2_1[:, :, :, int(W / 2):W]
        images_lv3_3 = images_lv2_2[:, :, :, 0:int(W / 2)]
        images_lv3_4 = images_lv2_2[:, :, :, int(W / 2):W]
        # B4
        images_lv4_1 = images_lv3_1[:, :, 0:int(H / 4), :]
        images_lv4_2 = images_lv3_1[:, :, int(H / 4):int(H / 2), :]
        images_lv4_3 = images_lv3_2[:, :, 0:int(H / 4), :]
        images_lv4_4 = images_lv3_2[:, :, int(H / 4):int(H / 2), :]
        images_lv4_5 = images_lv3_3[:, :, 0:int(H / 4), :]
        images_lv4_6 = images_lv3_3[:, :, int(H / 4):int(H / 2), :]
        images_lv4_7 = images_lv3_4[:, :, 0:int(H / 4), :]
        images_lv4_8 = images_lv3_4[:, :, int(H / 4):int(H / 2), :]

        # B4 encoder and recover orginal_size
        feature_lv4_1 = self.lrelu(self.conv_dila4(images_lv4_1))
        feature_lv4_2 = self.lrelu(self.conv_dila4(images_lv4_2))
        feature_lv4_3 = self.lrelu(self.conv_dila4(images_lv4_3))
        feature_lv4_4 = self.lrelu(self.conv_dila4(images_lv4_4))
        feature_lv4_5 = self.lrelu(self.conv_dila4(images_lv4_5))
        feature_lv4_6 = self.lrelu(self.conv_dila4(images_lv4_6))
        feature_lv4_7 = self.lrelu(self.conv_dila4(images_lv4_7))
        feature_lv4_8 = self.lrelu(self.conv_dila4(images_lv4_8))

        feature_lv4_top_left = torch.cat((feature_lv4_1, feature_lv4_2), 2)
        feature_lv4_top_right = torch.cat((feature_lv4_3, feature_lv4_4), 2)
        feature_lv4_bot_left = torch.cat((feature_lv4_5, feature_lv4_6), 2)
        feature_lv4_bot_right = torch.cat((feature_lv4_7, feature_lv4_8), 2)
        feature_lv4_top = torch.cat((feature_lv4_top_left, feature_lv4_top_right), 3)
        feature_lv4_bot = torch.cat((feature_lv4_bot_left, feature_lv4_bot_right), 3)
        feature_lv4 = torch.cat((feature_lv4_top, feature_lv4_bot), 2)

        # B3 encoder and recover orginal_size
        feature_lv3_1 = self.lrelu(self.conv_dila3(images_lv3_1))
        feature_lv3_2 = self.lrelu(self.conv_dila3(images_lv3_2))
        feature_lv3_3 = self.lrelu(self.conv_dila3(images_lv3_3))
        feature_lv3_4 = self.lrelu(self.conv_dila3(images_lv3_4))

        feature_lv3_top = torch.cat((feature_lv3_1, feature_lv3_2), 3)
        feature_lv3_bot = torch.cat((feature_lv3_3, feature_lv3_4), 3)
        feature_lv3 = torch.cat((feature_lv3_top, feature_lv3_bot), 2)

        # B2 encoder and recover orginal_size
        feature_lv2_1 = self.lrelu(self.conv_dila2(images_lv2_1))
        feature_lv2_2 = self.lrelu(self.conv_dila2(images_lv2_2))

        feature_lv2 = torch.cat((feature_lv2_1, feature_lv2_2), 2)

        # B1 encoder and recover orginal_size
        feature_lv1 = self.lrelu(self.conv_dila1(images_lv1))

        # aggregation, concatenation, reweighting
        feature_lv1 = feature_lv1 + feature_lv2 + feature_lv3 + feature_lv4
        feature_lv2 = feature_lv2 + feature_lv3 + feature_lv4
        feature_lv3 = feature_lv3 + feature_lv4
        feature_lv4 = feature_lv4
        feature_lv = torch.cat((feature_lv1, feature_lv2, feature_lv3, feature_lv4), 1)
        feature_lv = self.conv_first1(feature_lv)
        feature_lvv = self.lrelu(self.depth_conv1(feature_lv))
        feature_lvv = self.lrelu(self.depth_conv2(feature_lvv))
        feature_lvv = feature_lv * torch.sigmoid(feature_lvv)
        output = feature_lvv + in_fea
        return output



class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)



@ARCH_REGISTRY.register()
class Net(nn.Module):
    def __init__(self, channels=[32, 64, 128, 128], front_RBs=5, back_RBs=10, connection=False):
        super(Net, self).__init__()
        [ch1, ch2, ch3, ch4] = channels
        nf = ch2

        ResidualBlock_noBN_f = functools.partial(arch_util.ResidualBlock_noBN, nf=nf)
        self.conv_first_1 = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.conv_first_2 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)
        self.conv_first_3 = nn.Conv2d(nf, nf, 3, 2, 1, bias=True)

        self.feature_extraction = arch_util.make_layer(ResidualBlock_noBN_f, front_RBs)
        self.recon_trunk = arch_util.make_layer(ResidualBlock_noBN_f, back_RBs)

        self.upconv1 = nn.Conv2d(nf*2, nf * 4, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf*2, 64 * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.HRconv = nn.Conv2d(64*2, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.transformer = Encoder_patch66(d_model=1024, d_inner=2048, n_layers=6)
        # self.recon_trunk_light = arch_util.make_layer(ResidualBlock_noBN_f, 6)

        self.PPM1 = PPM(nf, nf//4, bins=(1,2,3,6))
        self.PPM2 = PPM(nf, nf//4, bins=(1,2,3,6))
        self.PPM3 = PPM(nf, nf//4, bins=(1,2,3,6))
        self.mcp = MCP(nf)
        self.conv_edge = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=True), nn.PReLU())
        self.conv_edge1 = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)
        self.conv_fre = nn.Sequential(nn.Conv2d(nf, nf, 3, 1, 1, bias=True), nn.PReLU())
        self.conv_fre1 = nn.Conv2d(nf, 3, 3, 1, 1, bias=True)

        self.fca_edge = eca_layer(channel=nf)
        self.fca_fre = eca_layer(channel=nf)



    def forward(self, x, mask, side_loss=False):

        x_center = x

        ### The encoder of our framework has three convolution layers (i.e., strides 1, 2, and 2) with one residual block after the encoder.

        L1_fea_1 = self.lrelu(self.conv_first_1(x_center)) # 512*960
        L1_fea_1 = self.PPM1(L1_fea_1)

        L1_fea_2 = self.lrelu(self.conv_first_2(L1_fea_1)) # downsample to 256*480
        L1_fea_2 = self.PPM2(L1_fea_2)

        L1_fea_3 = self.lrelu(self.conv_first_3(L1_fea_2)) # downsample to 128*240
        L1_fea_3 = self.PPM3(L1_fea_3)

        fea = self.feature_extraction(L1_fea_3)  # feature_extraction

        fea_light = self.mcp(fea)  # short-range branch
        feature_edge_m = self.conv_edge(fea_light)
        feature_edge = self.conv_edge1(feature_edge_m)

        feature_fre_m = self.conv_fre(fea_light)
        feature_fre = self.conv_fre1(feature_fre_m)


        ### Prepare mask for transformer
        h_feature = fea.shape[2]
        w_feature = fea.shape[3]
        mask = F.interpolate(mask, size=[h_feature, w_feature], mode='nearest') # resize the normalized SNR map

        xs = np.linspace(-1, 1, fea.size(3) // 4)  # 鐢ㄦ潵鍒涘缓绛夊樊鏁板垪
        ys = np.linspace(-1, 1, fea.size(2) // 4)
        xs = np.meshgrid(xs, ys)                   # 灏唜s涓瘡涓€涓暟鎹拰ys涓瘡涓€涓暟鎹粍鍚堢敓鎴愬緢澶氱偣,鐒跺悗灏嗚繖浜涚偣鐨剎鍧愭爣鏀惧叆鍒癤涓?y鍧愭爣鏀惧叆Y涓?骞朵笖鐩稿簲浣嶇疆鏄搴旂殑 xs*ys=60*32=1920
        xs = np.stack(xs, 2)
        xs = torch.Tensor(xs).unsqueeze(0).repeat(fea.size(0), 1, 1, 1).cuda()
        xs = xs.view(fea.size(0), -1, 2)


        ### SNR-aware transformer

        height = fea.shape[2]
        width = fea.shape[3]
        fea_unfold = F.unfold(fea, kernel_size=4, dilation=1, stride=4, padding=0)
        fea_unfold = fea_unfold.permute(0, 2, 1)


        mask_unfold = F.unfold(mask, kernel_size=4, dilation=1, stride=4, padding=0)  # unfold the mask
        mask_unfold = mask_unfold.permute(0, 2, 1)
        mask_unfold = torch.mean(mask_unfold, dim=2).unsqueeze(dim=-2)  # compute the average value in each patch
        mask_unfold[mask_unfold <= 0.5] = 0.0

        fea_unfold = self.transformer(fea_unfold, xs, src_mask=mask_unfold)
        fea_unfold = fea_unfold.permute(0, 2, 1)
        fea_unfold = nn.Fold(output_size=(height, width), kernel_size=(4, 4), stride=4, padding=0, dilation=1)(fea_unfold)


        ### SNR-based Spatially-varying Feature Fusion
        # channel = fea.shape[1]
        # mask = mask.repeat(1, channel, 1, 1)
        # fea = fea_unfold * (1 - mask) + fea_light * mask

        edge_attention = fea_unfold + feature_edge_m
        edge_attention = self.fca_edge(edge_attention)

        fre_attention = fea_unfold + feature_fre_m
        fre_attention = self.fca_fre(fre_attention)

        fea = fea_unfold + edge_attention + fre_attention

        ### decoder/ conv and skip connection and unsample by pixel_shuffle
        out_noise = self.recon_trunk(fea)
        out_noise = torch.cat([out_noise, L1_fea_3], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv1(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_2], dim=1)
        out_noise = self.lrelu(self.pixel_shuffle(self.upconv2(out_noise)))
        out_noise = torch.cat([out_noise, L1_fea_1], dim=1)
        out_noise = self.lrelu(self.HRconv(out_noise))
        out_noise = self.conv_last(out_noise)
        out_noise = out_noise + x_center

        return out_noise






class Fca_attention(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(Fca_attention, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        n, c, h, w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
            # If you have concerns about one-line-change, don't worry.   :)
            # In the ImageNet models, this line will never be triggered.
            # This is for compatibility in instance segmentation and object detection.
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MultiSpectralDCTLayer(nn.Module):
    """
    Generate dct filters
    """

    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # fixed random init
        # self.register_buffer('weight', torch.rand(channel, height, width))

        # learnable DCT init
        # self.register_parameter('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

        # learnable random init
        # self.register_parameter('weight', torch.rand(channel, height, width))

        # num_freq, h, w

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        # n, c, h, w = x.shape

        x = x * self.weight

        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


def get_freq_indices(method):
    assert method in ['top1', 'top2', 'top4', 'top8', 'top16', 'top32',
                      'bot1', 'bot2', 'bot4', 'bot8', 'bot16', 'bot32',
                      'low1', 'low2', 'low4', 'low8', 'low16', 'low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0, 0, 6, 0, 0, 1, 1, 4, 5, 1, 3, 0, 0, 0, 3, 2, 4, 6, 3, 5, 5, 2, 6, 5, 5, 3, 3, 4, 2, 2,
                             6, 1]
        all_top_indices_y = [0, 1, 0, 5, 2, 0, 2, 0, 0, 6, 0, 4, 6, 3, 5, 2, 6, 3, 3, 3, 5, 1, 1, 2, 4, 2, 1, 1, 3, 0,
                             5, 3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0, 0, 1, 1, 0, 2, 2, 1, 2, 0, 3, 4, 0, 1, 3, 0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1, 2,
                             3, 4]
        all_low_indices_y = [0, 1, 0, 1, 2, 0, 1, 2, 2, 3, 0, 0, 4, 3, 1, 5, 4, 3, 2, 1, 0, 6, 5, 4, 3, 2, 1, 0, 6, 5,
                             4, 3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6, 1, 3, 3, 2, 4, 1, 2, 4, 4, 5, 1, 4, 6, 2, 5, 6, 1, 6, 2, 2, 4, 3, 3, 5, 5, 6, 2, 5, 5,
                             3, 6]
        all_bot_indices_y = [6, 4, 4, 6, 6, 3, 1, 4, 4, 5, 6, 5, 2, 2, 5, 1, 4, 3, 5, 0, 3, 1, 1, 2, 4, 2, 1, 1, 5, 3,
                             3, 3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y