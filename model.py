from config import *
from efficientnet.model import mb_conv_block, round_filters, CONV_KERNEL_INITIALIZER, get_dropout
from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import utils

def fused_mb_conv_block(inputs, block_args: BlockArgs, activation='swish', drop_rate=None, prefix='', conv_dropout=None):
    """Fused Mobile Inverted Residual Bottleneck"""
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    Dropout = get_dropout(
        backend=backend,
        layers=layers,
        models=models,
        utils=utils
    )

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
    if conv_dropout and block_args.expand_ratio > 1:
        x = layers.Dropout(conv_dropout)(x)

    if has_se:
        pass

    # Output phase
    x = layers.Conv2D(block_args.output_filters,
                      kernel_size=1 if block_args.expand_ratio != 1 else block_args.kernel_size,
                      strides=1 if block_args.expand_ratio != 1 else block_args.strides,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      padding='same',
                      use_bias=False,
                      name=f'{prefix}project_conv')(x)

    x = layers.BatchNormalization(name=f'{prefix}project_bn')(x)
    if block_args.expand_ratio == 1:
        x = layers.Activation(activation=activation, name=f'{prefix}activation')(x)

    if all(s == 1 for s in block_args.strides) \
            and block_args.input_filters == block_args.output_filters:
        if drop_rate and drop_rate > 0:
            x = Dropout(drop_rate,
                        noise_shape=(None, 1, 1, 1),
                        name=f'{prefix}dropout')(x)
        x = layers.Add(name=f'{prefix}add')([x, inputs])
    return x


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
