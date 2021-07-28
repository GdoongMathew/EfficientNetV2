# Copyright 2019 The TensorFlow Authors, Tung Shu-Cheng. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import sys
sys.setrecursionlimit(1500)
import tensorflow as tf
from effnetv2_model import get_model
import efficientnetv2


def _port_block_weight(model_name: str, ori_block: tf.keras.layers.Layer, self_layers: list):
    bn_idx = 0
    conv_idx = 0
    depth_idx = 0
    se_idx = 0
    _weights = {
        w.name: wn for w, wn in zip(ori_block.weights, ori_block.get_weights())
    }

    for layer in self_layers:
        if not layer.weights:
            continue

        if 'bn' in layer.name:
            base_name = f'{model_name}/{ori_block.name}/tpu_batch_normalization' if not bn_idx \
                else f'{model_name}/{ori_block.name}/tpu_batch_normalization_{bn_idx}'

            bn_names = [
                f'{base_name}/gamma:0',
                f'{base_name}/beta:0',
                f'{base_name}/moving_mean:0',
                f'{base_name}/moving_variance:0',
            ]

            bn_weights = [_weights[name] for name in bn_names]
            layer.set_weights(bn_weights)
            bn_idx += 1

        elif 'dwconv' in layer.name:
            ori_name = f'{model_name}/{ori_block.name}/depthwise_conv2d/depthwise_kernel:0' if not depth_idx \
                else f'{model_name}/{ori_block.name}/depthwise_conv2d_{depth_idx}/depthwise_kernel:0'

            layer.set_weights([_weights[ori_name]])
            depth_idx += 1

        elif 'conv' in layer.name:
            ori_name = f'{model_name}/{ori_block.name}/conv2d/kernel:0' if not conv_idx \
                else f'{model_name}/{ori_block.name}/conv2d_{conv_idx}/kernel:0'

            layer.set_weights([_weights[ori_name]])
            conv_idx += 1

        elif 'se' in layer.name:

            if se_idx:
                ori_names = [
                    f'{model_name}/{ori_block.name}/se/conv2d_{se_idx}/kernel:0',
                    f'{model_name}/{ori_block.name}/se/conv2d_{se_idx}/bias:0',
                ]
            else:
                ori_names = [
                    f'{model_name}/{ori_block.name}/se/conv2d/kernel:0',
                    f'{model_name}/{ori_block.name}/se/conv2d/bias:0',
                ]

            se_weights = [_weights[name] for name in ori_names]
            layer.set_weights(se_weights)
            se_idx += 1


if __name__ == '__main__':

    from tqdm import tqdm

    name_map = {
        'EfficientNetV2_S': ['efficientnetv2-s', ('imagenet', 'imagenet21k', 'imagenet21k-ft1k')],
        'EfficientNetV2_M': ['efficientnetv2-m', ('imagenet', 'imagenet21k', 'imagenet21k-ft1k')],
        'EfficientNetV2_L': ['efficientnetv2-l', ('imagenet', 'imagenet21k', 'imagenet21k-ft1k')],
        'EfficientNetV2_XL': ['efficientnetv2-xl', ('imagenet21k', 'imagenet21k-ft1k')],
    }

    for target_model, ori in tqdm(name_map.items()):
        ori_name, pretrain_weights = ori
        for pretrain_weight in tqdm(pretrain_weights, leave=False):
            tf.keras.backend.clear_session()
            if '21k' in pretrain_weight and 'ft1k' not in pretrain_weight:
                classes = 21843
                include_top = False
            else:
                classes = 1000
                include_top = True
            ori_net = get_model(ori_name, weights=pretrain_weight, include_top=include_top)
            self_net = efficientnetv2.__getattribute__(target_model)(weights=None, include_top=include_top)
            layer_id = 1 # skip input layer
            # Stem
            stem_block = ori_net.get_layer('stem')
            stem_layers = []
            for self_l in self_net.layers[layer_id:]:
                if 'stem' not in self_l.name:
                    break
                stem_layers.append(self_l)
                layer_id += 1

            _port_block_weight(ori_name, stem_block, stem_layers)

            # MB blocks
            for block in ori_net.layers:
                if 'blocks' not in block.name:
                    break
                self_layers = []
                self_block_idx = self_net.layers[layer_id].name.split('block')[1].split('_')[0]

                for self_l in self_net.layers[layer_id:]:
                    if self_block_idx not in self_l.name:
                        break

                    self_layers.append(self_l)
                    layer_id += 1

                _port_block_weight(ori_name, block, self_layers)

            # Head block
            head_block = ori_net.get_layer('head')
            head_layers = []
            for self_l in self_net.layers[layer_id:]:
                head_layers.append(self_l)
                layer_id += 1

            _port_block_weight(ori_name, head_block, head_layers)

            if include_top:
                self_net.get_layer('probs').set_weights(ori_net.get_layer('dense').get_weights())
            self_net.save(f'{ori_name}_{pretrain_weight}.h5')
            tf.keras.backend.clear_session()
