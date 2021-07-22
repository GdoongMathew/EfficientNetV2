import string

from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils

from keras_applications.imagenet_utils import _obtain_input_shape

from .utils import CONV_KERNEL_INITIALIZER
from .utils import DENSE_KERNEL_INITIALIZER
from .utils import round_filters
from .utils import round_repeats
from .config import *


def get_dropout():
    """Wrapper over custom dropout. Fix problem of ``None`` shape for tf.keras.
    It is not possible to define FixedDropout class as global object,
    because we do not have modules for inheritance at first time.

    Issue:
        https://github.com/tensorflow/tensorflow/issues/30946
    """
    class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                           for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    return FixedDropout


def mb_conv_block(inputs,
                  block_args: BlockArgs,
                  activation='swish',
                  drop_rate=None,
                  prefix='',
                  conv_dropout=None,
                  mb_type='normal'):
    """Fused Mobile Inverted Residual Bottleneck"""
    assert mb_type in ['normal', 'fused']
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    Dropout = get_dropout()

    x = inputs

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = layers.Conv2D(filters,
                          1 if mb_type == 'normal' else block_args.kernel_size,
                          strides=1 if mb_type == 'normal' else block_args.strides,
                          kernel_initializer=CONV_KERNEL_INITIALIZER,
                          padding='same',
                          use_bias=False,
                          name=f'{prefix}expand_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=f'{prefix}expand_bn')(x)
        x = layers.Activation(activation=activation, name=f'{prefix}_expand_activation')(x)

    if mb_type is 'normal':
        x = layers.DepthwiseConv2D(block_args.kernel_size,
                                   block_args.strides,
                                   depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                   padding='same',
                                   use_bias=False,
                                   name=f'{prefix}dwconv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=f'{prefix}bn')(x)
        x = layers.Activation(activation=activation, name=f'{prefix}activation')(x)

    if conv_dropout and block_args.expand_ratio > 1:
        x = layers.Dropout(conv_dropout)(x)

    if has_se:
        num_reduced_filters = max(1, int(
            block_args.input_filters * block_args.se_ratio
        ))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, filters) if backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        se_tensor = layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = layers.Conv2D(num_reduced_filters, 1,
                                  activation=activation,
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters, 1,
                                  activation='sigmoid',
                                  padding='same',
                                  use_bias=True,
                                  kernel_initializer=CONV_KERNEL_INITIALIZER,
                                  name=prefix + 'se_expand')(se_tensor)
        if backend.backend() == 'theano':
            # For the Theano backend, we have to explicitly make
            # the excitation weights broadcastable.
            pattern = ([True, True, True, False] if backend.image_data_format() == 'channels_last'
                       else [True, False, True, True])
            se_tensor = layers.Lambda(
                lambda x: backend.pattern_broadcast(x, pattern),
                name=prefix + 'se_broadcast')(se_tensor)
        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = layers.Conv2D(block_args.output_filters,
                      kernel_size=1 if block_args.expand_ratio != 1 else block_args.kernel_size,
                      strides=1 if block_args.expand_ratio != 1 else block_args.strides,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      padding='same',
                      use_bias=False,
                      name=f'{prefix}project_conv')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=f'{prefix}project_bn')(x)
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


def EfficientNetV2(blocks_args,
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

    assert isinstance(blocks_args, list) and False not in [isinstance(block_args, BlockArgs) for block_args in blocks_args]

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=default_resolution,
                                      min_size=32,
                                      data_format=backend.image_data_format(),
                                      require_flatten=include_top,
                                      weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if backend.backend() == 'tensorflow':
            from tensorflow.keras.backend import is_keras_tensor
        else:
            is_keras_tensor = backend.is_keras_tensor
        if not is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    # build stem layer
    x = img_input

    x = layers.Conv2D(round_filters(blocks_args[0].input_filters, width_coefficient, depth_divisor), 3,
                      strides=2,
                      kernel_initializer=CONV_KERNEL_INITIALIZER,
                      padding='same',
                      name='stem_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='stem_bn')(x)
    x = layers.Activation(activation=activation)(x)

    mb_type = {
        0: 'normal',
        1: 'fused'
    }

    num_blocks_total = sum(block_args.num_repeat for block_args in blocks_args)
    block_num = 0

    # build blocks
    for idx, block_args in enumerate(blocks_args):
        assert isinstance(block_args, BlockArgs)
        assert block_args.num_repeat > 0
        input_filters = round_filters(block_args.input_filters,
                                      width_coefficient,
                                      depth_divisor)
        output_filters = round_filters(block_args.output_filters,
                                       width_coefficient,
                                       depth_divisor)
        repeats = round_repeats(block_args.num_repeat, depth_coefficient)

        block_args = block_args._replace(
            input_filters=input_filters,
            output_filters=output_filters,
            num_repeat=repeats
        )
        drop_rate = dropout_rate * float(block_num) / num_blocks_total

        conv_type = mb_type[block_args.conv_type]
        x = mb_conv_block(x, block_args,
                          activation=activation,
                          drop_rate=drop_rate,
                          mb_type=conv_type,
                          prefix=f'block{idx + 1}a_')
        block_num += 1
        if block_args.num_repeat > 1:
            block_args = block_args._replace(
                input_filters=block_args.output_filters,
                strides=[1, 1]
            )
            for _idx in range(block_args.num_repeat - 1):
                drop_rate = dropout_rate * float(block_num) / num_blocks_total
                block_prefix = f'block{idx + 1}{string.ascii_lowercase[_idx + 1]}_'
                x = mb_conv_block(x, block_args,
                                  activation=activation,
                                  drop_rate=drop_rate,
                                  mb_type=conv_type,
                                  prefix=block_prefix)
                block_num += 1

    # build head
    x = layers.Conv2D(
        filters=round_filters(1280, width_coefficient, depth_divisor),
        kernel_size=1,
        strides=1,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        padding='same',
        use_bias=False,
        name='head_conv')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='head_bn')(x)
    x = layers.Activation(activation=activation, name='head_activation')(x)
    if pooling == 'avg':
        x = layers.GlobalAveragePooling2D(name='head_avg_pool')(x)
    elif pooling == 'max':
        x = layers.GlobalMaxPooling2D(name='head_max_pool')(x)
    if dropout_rate and dropout_rate > 0:
        x = layers.Dropout(dropout_rate, name='head_dropout')(x)

    if include_top:
        x = layers.Dense(classes,
                         activation='softmax',
                         kernel_initializer=DENSE_KERNEL_INITIALIZER,
                         name='probs')(x)

    inputs = img_input if input_tensor is None else keras_utils.get_source_inputs(input_tensor)

    model = models.Model(inputs, x, name=model_name)

    if weights:
        model.load_weights(weights)
    return model


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
        V2_XL_BLOCKS_ARGS,
        1., 1., 384,
        dropout_rate=0.4,
        model_name='efficientnetv2_xl',
        include_top=include_top,
        weights=weights,
        input_tensor=input_tensor,
        input_shape=input_shape,
        pooling=pooling,
        classes=classes,
        **kwargs
    )
