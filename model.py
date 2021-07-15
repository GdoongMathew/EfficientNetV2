from config import *


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
                   pooling=None,
                   classes=1000,
                   **kwargs):
    pass


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
