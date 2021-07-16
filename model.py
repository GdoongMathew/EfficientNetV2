from config import *
from efficientnet.model import mb_conv_block, round_filters, CONV_KERNEL_INITIALIZER
from tensorflow.keras import layers
from tensorflow.keras import backend


def fused_mb_conv_block(inputs, block_args: BlockArgs, activation='swish', drop_rate=None, prefix=''):
    """Fused Mobile Inverted Residual Bottleneck"""
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = inputs

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = layers.Conv2D(filters,
                          block_args.kernel_size,
                          strides=block_args.strides,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          padding='same',
                          use_bias=False,
                          name=f'{prefix}expand_conv')(x)
        x = layers.BatchNormalization(name=f'{prefix}expand_bn')(x)
        x = layers.Activation(activation=activation, name=f'{prefix}_expand_activation')(x)


def EfficientNetV2(block_args,
                   width_coefficient,
                   depth_coefficient,
                   default_resolution,
                   dropout_rate=0.2,
                   depth_divisor=8,
                   model_name='efficientnetv2',
                   include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   activation='swish',
                   pooling=None,
                   classes=1000,
                   **kwargs):

    assert isinstance(block_args, list) and False not in [isinstance(bkg_arg, BlockArgs) for bkg_arg in block_args]

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        img_input = input_tensor

    # build stem layer
    x = img_input

    x = layers.Conv2D(round_filters(block_args[0].input_filters, width_coefficient, depth_divisor), 3,
                      strides=2,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      padding='same',
                      name='stem_conv')(x)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.Activation(activation=activation)(x)

    # build blocks


def EfficientNetV2_Base(include_top=True,
                        weights='imagenet',
                        input_tensor=None,
                        input_shape=None,
                        pooling=None,
                        classes=1000,
                        **kwargs
                        ):
    return EfficientNetV2(
        V2_BASE_BLOCKS_ARGS,
        1., 1., 300,
        dropout_rate=0.2,
        model_name='efficientnetv2_base',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        **kwargs
    )


def EfficientNetV2_S(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     classes=1000,
                     **kwargs
                     ):
    return EfficientNetV2(
        V2_S_BLOCKS_ARGS,
        1., 1., 300,
        dropout_rate=0.2,
        model_name='efficientnetv2_s',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        **kwargs
    )


def EfficientNetV2_M(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     classes=1000,
                     **kwargs
                     ):
    return EfficientNetV2(
        V2_M_BLOCKS_ARGS,
        1., 1., 384,
        dropout_rate=0.2,
        model_name='efficientnetv2_m',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        **kwargs
    )


def EfficientNetV2_L(include_top=True,
                     weights='imagenet',
                     input_tensor=None,
                     input_shape=None,
                     pooling=None,
                     classes=1000,
                     **kwargs
                     ):
    return EfficientNetV2(
        V2_L_BLOCKS_ARGS,
        1., 1., 384,
        dropout_rate=0.4,
        model_name='efficientnetv2_l',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        **kwargs
    )


def EfficientNetV2_XL(include_top=True,
                      weights='imagenet',
                      input_tensor=None,
                      input_shape=None,
                      pooling=None,
                      classes=1000,
                      **kwargs
                      ):
    return EfficientNetV2(
        V2_L_BLOCKS_ARGS,
        1., 1., 384,
        dropout_rate=0.4,
        model_name='efficientnetv2_l',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        **kwargs
    )
