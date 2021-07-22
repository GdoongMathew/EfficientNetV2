import collections

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type'
])
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

V2_BASE_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=1, input_filters=32, output_filters=16,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=16, output_filters=32,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=32, output_filters=48,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=48, output_filters=96,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=5, input_filters=96, output_filters=112,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=8, input_filters=112, output_filters=192,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
]

V2_S_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=2, input_filters=24, output_filters=24,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=4, input_filters=24, output_filters=48,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=4, input_filters=48, output_filters=64,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=6, input_filters=64, output_filters=128,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=9, input_filters=112, output_filters=160,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=15, input_filters=160, output_filters=256,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
]

V2_M_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=3, input_filters=24, output_filters=24,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=5, input_filters=24, output_filters=48,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=5, input_filters=48, output_filters=80,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=7, input_filters=80, output_filters=160,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=14, input_filters=160, output_filters=176,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=18, input_filters=176, output_filters=384,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=5, input_filters=304, output_filters=512,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
]

V2_L_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=4, input_filters=32, output_filters=32,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=7, input_filters=32, output_filters=64,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=7, input_filters=64, output_filters=96,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=10, input_filters=96, output_filters=192,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=19, input_filters=192, output_filters=224,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=25, input_filters=224, output_filters=384,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=7, input_filters=384, output_filters=640,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
]

V2_XL_BLOCKS_ARGS = [
    BlockArgs(kernel_size=3, num_repeat=4, input_filters=32, output_filters=32,
              expand_ratio=1, id_skip=True, strides=[1, 1], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=8, input_filters=32, output_filters=64,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=8, input_filters=64, output_filters=96,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=None, conv_type=1),
    BlockArgs(kernel_size=3, num_repeat=16, input_filters=96, output_filters=192,
              expand_ratio=4, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=24, input_filters=192, output_filters=256,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=32, input_filters=256, output_filters=512,
              expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0),
    BlockArgs(kernel_size=3, num_repeat=8, input_filters=512, output_filters=640,
              expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0),
]
