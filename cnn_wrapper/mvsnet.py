#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
MVSNet sub-models.
"""

from cnn_wrapper.network import Network


########################################################################################
############################# 2D feature extraction nework #############################
########################################################################################

class Pix2PixUpsample4(Network):
    """
    pix2pix network architecture, original, 4*input_size=output_size
    """

    # def setup(self):
    #     print('pix2pix network architecture, original, 4*input_size=output_size, base_filter=8')
    #     base_filter = 8
    #     (self.feed('data')
    #      # .conv(4, base_filter, 2, name='conv0')
    #      # .conv_bn(4, base_filter * 2, 2, center=True, scale=True, name='conv1')
    #      .conv_bn(4, base_filter * 4, 2, center=True, scale=True, name='conv2')
    #      .conv_bn(4, base_filter * 8, 2, center=True, scale=True, name='conv3')
    #      .conv_bn(4, base_filter * 8, 2, center=True, scale=True, name='conv4')
    #      # .conv_bn(4, base_filter * 8, 2, center=True, scale=True, name='conv5')
    #      # .conv_bn(4, base_filter * 8, 2, center=True, scale=True, name='conv6')
    #      .deconv_bn(4, base_filter * 8, 2, center=True, scale=True, name='dconv6'))
    #
    #     # (self.feed('dconv6', 'conv5')
    #     #  .concat(axis=-1, name='concat5')
    #     #  .deconv_bn(4, base_filter * 8, 2, center=True, scale=True, name='dconv5'))
    #     #
    #     # (self.feed('dconv5', 'conv4')
    #     #  .concat(axis=-1, name='concat4')
    #     #  .deconv_bn(4, base_filter * 8, 2, center=True, scale=True, name='dconv4'))
    #
    #     (self.feed('dconv6', 'conv3')
    #      .concat(axis=-1, name='concat3')
    #      .deconv_bn(4, base_filter * 4, 2, center=True, scale=True, name='dconv3'))
    #
    #     (self.feed('dconv3', 'conv2')
    #      .concat(axis=-1, name='concat2')
    #      .deconv_bn(4, base_filter * 2, 2, center=True, scale=True, name='dconv2')
    #      .deconv_bn(4, base_filter * 1, 2, center=True, scale=True, name='dconv1')
    #      .deconv(4, 3, 2, name='dconv0', relu=False)
    #      .tanh(name='out_image')
    #      )


    def setup(self):
        print('pix2pix network architecture, original, 4*input_size=output_size, base_filter=8')
        base_filter = 32
        (self.feed('data')
         # .conv(4, base_filter, 2, name='conv0')
         # .conv_bn(4, base_filter * 2, 2, center=True, scale=True, name='conv1')
         .conv_bn(4, base_filter * 4, 2, center=True, scale=True, name='conv2')
         .conv_bn(4, base_filter * 8, 2, center=True, scale=True, name='conv3')
         .conv_bn(4, base_filter * 8, 2, center=True, scale=True, name='conv4')
         .conv_bn(4, base_filter * 8, 2, center=True, scale=True, name='conv5')
         .conv_bn(4, base_filter * 8, 2, center=True, scale=True, name='conv6')
         .deconv_bn(4, base_filter * 8, 2, center=True, scale=True, name='dconv6'))

        (self.feed('dconv6', 'conv5')
         .concat(axis=-1, name='concat5')
         .deconv_bn(4, base_filter * 8, 2, center=True, scale=True, name='dconv5'))

        (self.feed('dconv5', 'conv4')
         .concat(axis=-1, name='concat4')
         .deconv_bn(4, base_filter * 8, 2, center=True, scale=True, name='dconv4'))

        (self.feed('dconv4', 'conv3')
         .concat(axis=-1, name='concat3')
         .deconv_bn(4, base_filter * 4, 2, center=True, scale=True, name='dconv3'))

        (self.feed('dconv3', 'conv2')
         .concat(axis=-1, name='concat2')
         .deconv_bn(4, base_filter * 2, 2, center=True, scale=True, name='dconv2')
         .deconv_bn(4, base_filter * 1, 2, center=True, scale=True, name='dconv1')
         .deconv(4, 3, 2, name='dconv0', relu=False)
         .tanh(name='out_image')
         )



class SNetDS2BN_base_8(Network):
    """2D U-Net with group normalization."""

    def setup(self):
        print ('2D SNet with 32 channel output')
        base_filter = 8
        (self.feed('data')
         .conv_bn(3,base_filter,1,dilation_rate=1,center=True,scale=True,name="sconv0_0")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True, name="sconv0_1")
         .conv_bn(3, base_filter*2, 1,dilation_rate=2, center=True, scale=True, name="sconv0_2")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv0_3")
         )
        (self.feed('sconv0_2')
         .conv_bn(3, base_filter*2, 1,dilation_rate=3, center=True, scale=True, name="sconv1_2")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv1_3")
         )
        (self.feed('sconv0_2')
        .conv_bn(3, base_filter*2, 1,dilation_rate=4, center=True, scale=True, name="sconv2_2")
        .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv2_3")
        )
        (self.feed('sconv0_3','sconv1_3','sconv2_3')
        .concat(axis=-1,name='sconcat')
        # .convs_bn(3,base_filter*2,1,dilation_rate=1, center=True, scale=True,relu=True, name='sconv3_0')
        .conv(3,base_filter*2,1,relu=False,name='sconv3_0')
        )


class SNetDS2BN_base_8_ms(Network):
    """2D U-Net with group normalization."""

    def setup(self):
        print ('2D SNet with 32 channel output')
        base_filter = 8
        (self.feed('data')
         .conv_bn(3,base_filter,1,dilation_rate=1,center=True,scale=True,name="sconv0_0")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True, name="sconv0_1")
         .conv_bn(3, base_filter*2, 1,dilation_rate=2, center=True, scale=True, name="sconv0_2")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv0_3")
         )
        (self.feed('sconv0_2')
         .conv_bn(3, base_filter*2, 1,dilation_rate=3, center=True, scale=True, name="sconv1_2")
         .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv1_3")
         )
        (self.feed('sconv0_2')
        .conv_bn(3, base_filter*2, 1,dilation_rate=4, center=True, scale=True, name="sconv2_2")
        .conv_bn(3, base_filter*2, 1,dilation_rate=1, center=True, scale=True,relu=True, name="sconv2_3")
        )
        (self.feed('sconv0_0', 'sconv0_3','sconv1_3','sconv2_3')
        .concat(axis=-1,name='sconcat')
        # .convs_bn(3,base_filter*2,1,dilation_rate=1, center=True, scale=True,relu=True, name='sconv3_0')
        .conv(3,base_filter*2,1,relu=False,name='sconv3_0')
        )



class UniNetDS2(Network):
    """Simple UniNet, as described in the paper."""

    def setup(self):
        print ('2D with 32 filters')
        base_filter = 4
        (self.feed('data')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='conv0_0')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='conv0_1')
        .conv_bn(5, base_filter * 2, 2, center=True, scale=True, name='conv1_0')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_1')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_2')
        .conv_bn(5, base_filter * 4, 2, center=True, scale=True, name='conv2_0')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='conv2_1')
        .conv(3, base_filter * 4, 1, biased=False, relu=False, name='conv2_2'))


class UniNetDS1(Network):
    """Simple UniNet, as described in the paper."""

    def setup(self):
        print ('2D with 32 filters')
        base_filter = 4
        (self.feed('data')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='conv0_0')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='conv0_1')
        .conv_bn(5, base_filter * 2, 1, center=True, scale=True, name='conv1_0')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_1')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_2')
        .conv_bn(5, base_filter * 4, 1, center=True, scale=True, name='conv2_0')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='conv2_1')
        .conv(3, base_filter * 4, 1, biased=False, relu=False, name='conv2_2'))

class UniNetDS2GN(Network):
    """Simple UniNet with group normalization."""

    def setup(self):
        print ('2D with 32 filters')
        base_filter = 8
        (self.feed('data')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='conv0_0')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='conv0_1')
        .conv_gn(5, base_filter * 2, 2, center=True, scale=True, name='conv1_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_1')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='conv1_2')
        .conv_gn(5, base_filter * 4, 2, center=True, scale=True, name='conv2_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='conv2_1')
        .conv(3, base_filter * 4, 1, biased=False, relu=False, name='conv2_2'))

class UNetDS2GN(Network):
    """2D U-Net with group normalization."""

    def setup(self):
        print ('2D UNet with 32 channel output')
        base_filter = 8
        (self.feed('data')
        .conv_gn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv1_0')
        .conv_gn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv2_0')
        .conv_gn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv3_0')
        .conv_gn(3, base_filter * 16, 2, center=True, scale=True, name='2dconv4_0'))

        (self.feed('data')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv0_1')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv0_2'))

        (self.feed('2dconv1_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_1')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv1_2'))

        (self.feed('2dconv2_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_1')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv2_2'))

        (self.feed('2dconv3_0')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_1')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv3_2'))

        (self.feed('2dconv4_0')
        .conv_gn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_1')
        .conv_gn(3, base_filter * 16, 1, center=True, scale=True, name='2dconv4_2')
        .deconv_gn(3, base_filter * 8, 2, center=True, scale=True, name='2dconv5_0'))

        (self.feed('2dconv5_0', '2dconv3_2')
        .concat(axis=-1, name='2dconcat5_0')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_1')
        .conv_gn(3, base_filter * 8, 1, center=True, scale=True, name='2dconv5_2')
        .deconv_gn(3, base_filter * 4, 2, center=True, scale=True, name='2dconv6_0'))

        (self.feed('2dconv6_0', '2dconv2_2')
        .concat(axis=-1, name='2dconcat6_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_1')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='2dconv6_2')
        .deconv_gn(3, base_filter * 2, 2, center=True, scale=True, name='2dconv7_0'))

        (self.feed('2dconv7_0', '2dconv1_2')
        .concat(axis=-1, name='2dconcat7_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv7_1')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='2dconv7_2')
        .deconv_gn(3, base_filter, 2, center=True, scale=True, name='2dconv8_0'))

        (self.feed('2dconv8_0', '2dconv0_2')
        .concat(axis=-1, name='2dconcat8_0')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv8_1')
        .conv_gn(3, base_filter, 1, center=True, scale=True, name='2dconv8_2')   # end of UNet
        .conv_gn(5, base_filter * 2, 2, center=True, scale=True, name='conv9_0')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='conv9_1')
        .conv_gn(3, base_filter * 2, 1, center=True, scale=True, name='conv9_2')
        .conv_gn(5, base_filter * 4, 2, center=True, scale=True, name='conv10_0')
        .conv_gn(3, base_filter * 4, 1, center=True, scale=True, name='conv10_1')
        .conv(3, base_filter * 4, 1, biased=False, relu = False, name='conv10_2'))

########################################################################################
###################### 3D CNNs cost volume regularization network ######################
########################################################################################

class RegNetUS0(Network):
    """network for regularizing 3D cost volume in a encoder-decoder style. Keeping original size."""

    def setup(self):
        print ('Shallow 3D UNet with 8 channel input')
        base_filter = 8
        (self.feed('data')
        .conv_bn(3, base_filter * 2, 2, center=True, scale=True, name='3dconv1_0')
        .conv_bn(3, base_filter * 4, 2, center=True, scale=True, name='3dconv2_0')
        .conv_bn(3, base_filter * 8, 2, center=True, scale=True, name='3dconv3_0'))

        (self.feed('data')
        .conv_bn(3, base_filter, 1, center=True, scale=True, name='3dconv0_1'))

        (self.feed('3dconv1_0')
        .conv_bn(3, base_filter * 2, 1, center=True, scale=True, name='3dconv1_1'))

        (self.feed('3dconv2_0')
        .conv_bn(3, base_filter * 4, 1, center=True, scale=True, name='3dconv2_1'))

        (self.feed('3dconv3_0')
        .conv_bn(3, base_filter * 8, 1, center=True, scale=True, name='3dconv3_1')
        .deconv_bn(3, base_filter * 4, 2, center=True, scale=True, name='3dconv4_0'))

        (self.feed('3dconv4_0', '3dconv2_1')
        .add(name='3dconv4_1')
        .deconv_bn(3, base_filter * 2, 2, center=True, scale=True, name='3dconv5_0'))

        (self.feed('3dconv5_0', '3dconv1_1')
        .add(name='3dconv5_1')
        .deconv_bn(3, base_filter, 2, center=True, scale=True, name='3dconv6_0'))

        (self.feed('3dconv6_0', '3dconv0_1')
        .add(name='3dconv6_1')
        .conv(3, 1, 1, biased=False, relu=False, name='3dconv6_2'))

class RefineNet(Network):
    """network for depth map refinement using original image."""

    def setup(self):

        (self.feed('color_image', 'depth_image')
        .concat(axis=3, name='concat_image'))

        (self.feed('concat_image')
        .conv_bn(3, 32, 1, name='refine_conv0')
        .conv_bn(3, 32, 1, name='refine_conv1')
        .conv_bn(3, 32, 1, name='refine_conv2')
        .conv(3, 1, 1, relu=False, name='refine_conv3'))

        (self.feed('refine_conv3', 'depth_image')
        .add(name='refined_depth_image'))
    
