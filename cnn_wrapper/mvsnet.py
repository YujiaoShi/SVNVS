#!/usr/bin/env python
"""
Copyright 2019, Yao Yao, HKUST.
MVSNet sub-models.
"""

from cnn_wrapper.network import Network


########################################################################################
############################# 2D feature extraction nework #############################
########################################################################################

class SNetDS2BN_base_8(Network):
    """2D U-Net with group normalization."""

    def setup(self):
        print ('2D SNet with 16 channel output')
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
        print ('2D SNet with 16 channel output')
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


