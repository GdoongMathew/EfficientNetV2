IMAGENET_WEIGHTS_URL = 'https://github.com/GdoongMathew/EfficientNetV2/releases/download/v0.0.1/'

WEIGHTS_MAP = {
    'efficientnetv2_s': {
        'imagenet':         {
            True:   'efficientnetv2-s_imagenet_top.h5',
            False:  'efficientnetv2-s_imagenet_notop.h5',
        },
        'imagenet21k':      {
            False:  'efficientnetv2-s_imagenet21k_notop.h5',
        },
        'imagenet21k-ft1k': {
            True:   'efficientnetv2-s_imagenet21k-ft1k_top.h5',
            False:  'efficientnetv2-s_imagenet21k-ft1k_notop.h5',
        },
    },
    'efficientnetv2_m': {
        'imagenet':         {
            True:   'efficientnetv2-m_imagenet_top.h5',
            False:  'efficientnetv2-m_imagenet_notop.h5',
        },
        'imagenet21k':      {
            False:  'efficientnetv2-m_imagenet21k_notop.h5',
        },
        'imagenet21k-ft1k': {
            True:   'efficientnetv2-m_imagenet21k-ft1k_top.h5',
            False:  'efficientnetv2-m_imagenet21k-ft1k_notop.h5',
        },
    },
    'efficientnetv2_l': {
        'imagenet':         {
            True:   'efficientnetv2-l_imagenet_top.h5',
            False:  'efficientnetv2-l_imagenet_notop.h5',
        },
        'imagenet21k':      {
            False:  'efficientnetv2-l_imagenet21k_notop.h5',
        },
        'imagenet21k-ft1k': {
            True:   'efficientnetv2-l_imagenet21k-ft1k_top.h5',
            False:  'efficientnetv2-l_imagenet21k-ft1k_notop.h5',
        },
    },
    'efficientnetv2_xl': {
        'imagenet21k':      {
            False:  'efficientnetv2-xl_imagenet21k_notop.h5',
        },
        'imagenet21k-ft1k': {
            True:   'efficientnetv2-xl_imagenet21k-ft1k_top.h5',
            False:  'efficientnetv2-xl_imagenet21k-ft1k_notop.h5',
        },
    }
}
