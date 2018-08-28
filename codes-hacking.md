# Object-Detection-API
2018-08-13

## python path
export PYTHONPATH=`pwd`:`pwd`/slim

## ssd-mnet2
python object_detection/model_main.py --pipeline_config_path=../training/20180813/models/ssdlite_mobilenet_v2_wider.config --model_dir=../training/20180813/models/train/  --num_train_steps=1000 --num_eval_steps=100000 --alsologtostderr

## ssd-mnet2
python object_detection/model_main.py --pipeline_config_path=../training/20180813/models/ssd_mobilenet_v1_wider.config --model_dir=../training/20180813/models/train/  --num_train_steps=100000 --num_eval_steps=1000 --alsologtostderr



# object_detection/model_main.py

def main(unused_argv):
  
  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      eval_steps=FLAGS.num_eval_steps)


# object_detection/model_lib.py

def create_estimator_and_inputs(run_config,
                                hparams,    ## [('load_pretrained', True)]
                                pipeline_config_path,
                                train_steps=None,
                                eval_steps=None,
                                model_fn_creator=create_model_fn,
                                use_tpu_estimator=False,
                                use_tpu=False,
                                num_shards=1,
                                params=None,
                                **kwargs):


  detection_model_fn = functools.partial(
      model_builder.build, model_config=model_config)


  model_fn = model_fn_creator(detection_model_fn, configs, hparams, use_tpu)
  if use_tpu_estimator:
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        train_batch_size=train_config.batch_size,
        # For each core, only batch size 1 is supported for eval.
        eval_batch_size=num_shards * 1 if use_tpu else 1,
        use_tpu=use_tpu,
        config=run_config,
        # TODO(lzc): Remove conditional after CMLE moves to TF 1.9
        params=params if params else {})
  else:
    estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config)



def create_model_fn(detection_model_fn, configs, hparams, use_tpu=False):

  def model_fn(features, labels, mode, params=None):

    detection_model = detection_model_fn(is_training=is_training,
                                         add_summaries=(not use_tpu))


  return model_fn


# object_detection/builders/model_builder.py

def _build_ssd_model(ssd_config, is_training, add_summaries,
                     add_background_class=True):

  # Feature extractor
  feature_extractor = _build_ssd_feature_extractor(
      feature_extractor_config=ssd_config.feature_extractor,
      is_training=is_training)

  return ssd_meta_arch.SSDMetaArch(
      is_training,
      anchor_generator,
      ssd_box_predictor,
      box_coder,
      feature_extractor,
      matcher,
      region_similarity_calculator,
      encode_background_as_zeros,
      negative_class_weight,
      image_resizer_fn,
      non_max_suppression_fn,
      score_conversion_fn,
      classification_loss,
      localization_loss,
      classification_weight,
      localization_weight,
      normalize_loss_by_num_matches,
      hard_example_miner,
      target_assigner_instance=target_assigner_instance,
      add_summaries=add_summaries,
      normalize_loc_loss_by_codesize=normalize_loc_loss_by_codesize,
      freeze_batchnorm=ssd_config.freeze_batchnorm,
      inplace_batchnorm_update=ssd_config.inplace_batchnorm_update,
      add_background_class=add_background_class,
      random_example_sampler=random_example_sampler,
      expected_classification_loss_under_sampling=
      expected_classification_loss_under_sampling)



# models/ssd_mobilenet_v1_feature_extractor.py

class SSDMobileNetV1FeatureExtractor(ssd_meta_arch.SSDFeatureExtractor):


  def extract_features(self, preprocessed_inputs):
    """Extract features from preprocessed inputs.

    Args:
      preprocessed_inputs: a [batch, height, width, channels] float tensor
        representing a batch of images.

    Returns:
      feature_maps: a list of tensors where the ith tensor has shape
        [batch, height_i, width_i, depth_i]
    """
    preprocessed_inputs = shape_utils.check_min_image_dim(
        33, preprocessed_inputs)

    feature_map_layout = {
        'from_layer': ['Conv2d_11_pointwise', 'Conv2d_13_pointwise', '', '',
                       '', ''],
        'layer_depth': [-1, -1, 512, 256, 256, 128],
        'use_explicit_padding': self._use_explicit_padding,
        'use_depthwise': self._use_depthwise,
    }

    with tf.variable_scope('MobilenetV1',
                           reuse=self._reuse_weights) as scope:
      with slim.arg_scope(
          mobilenet_v1.mobilenet_v1_arg_scope(
              is_training=None, regularize_depthwise=True)):
        #print "_override_base_feature_extractor_hyperparams= ", self._override_base_feature_extractor_hyperparams
        #print "context_manager.IdentityContextManager()= ", context_manager.IdentityContextManager()
        #print "_conv_hyperparams_fn()", self._conv_hyperparams_fn()
        #with (slim.arg_scope(self._conv_hyperparams_fn())
        #      if self._override_base_feature_extractor_hyperparams
        #      else context_manager.IdentityContextManager()):
        with (slim.arg_scope(self._conv_hyperparams_fn())
              if False
              else context_manager.IdentityContextManager()):
          _, image_features = mobilenet_v1.mobilenet_v1_base(
              ops.pad_to_multiple(preprocessed_inputs, self._pad_to_multiple),
              final_endpoint='Conv2d_13_pointwise',
              min_depth=self._min_depth,
              depth_multiplier=self._depth_multiplier,
              use_explicit_padding=self._use_explicit_padding,
              scope=scope)

      #print "image_features"
      #for k, v in image_features.items():
      #  print k, v

      with slim.arg_scope(self._conv_hyperparams_fn()):
        feature_maps = feature_map_generators.multi_resolution_feature_maps(
            feature_map_layout=feature_map_layout,
            depth_multiplier=self._depth_multiplier,
            min_depth=self._min_depth,
            insert_1x1_conv=True,
            image_features=image_features)

      #print "feature_maps"
      #for k, v in feature_maps.items():
      #  print k, v

    return feature_maps.values()




# object_detection/models/feature_map_generators.py


def multi_resolution_feature_maps(feature_map_layout, depth_multiplier,
                                  min_depth, insert_1x1_conv, image_features):
  """Generates multi resolution feature maps from input image features.

  Generates multi-scale feature maps for detection as in the SSD papers by
  Liu et al: https://arxiv.org/pdf/1512.02325v2.pdf, See Sec 2.1.

  More specifically, it performs the following two tasks:
  1) If a layer name is provided in the configuration, returns that layer as a
     feature map.
  2) If a layer name is left as an empty string, constructs a new feature map
     based on the spatial shape and depth configuration. Note that the current
     implementation only supports generating new layers using convolution of
     stride 2 resulting in a spatial resolution reduction by a factor of 2.
     By default convolution kernel size is set to 3, and it can be customized
     by caller.

  An example of the configuration for Inception V3:
  {
    'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', ''],
    'layer_depth': [-1, -1, -1, 512, 256, 128]
  }

  Args:
    feature_map_layout: Dictionary of specifications for the feature map
      layouts in the following format (Inception V2/V3 respectively):
      {
        'from_layer': ['Mixed_3c', 'Mixed_4c', 'Mixed_5c', '', '', ''],
        'layer_depth': [-1, -1, -1, 512, 256, 128]
      }
      or
      {
        'from_layer': ['Mixed_5d', 'Mixed_6e', 'Mixed_7c', '', '', '', ''],
        'layer_depth': [-1, -1, -1, 512, 256, 128]
      }
      If 'from_layer' is specified, the specified feature map is directly used
      as a box predictor layer, and the layer_depth is directly infered from the
      feature map (instead of using the provided 'layer_depth' parameter). In
      this case, our convention is to set 'layer_depth' to -1 for clarity.
      Otherwise, if 'from_layer' is an empty string, then the box predictor
      layer will be built from the previous layer using convolution operations.
      Note that the current implementation only supports generating new layers
      using convolutions of stride 2 (resulting in a spatial resolution
      reduction by a factor of 2), and will be extended to a more flexible
      design. Convolution kernel size is set to 3 by default, and can be
      customized by 'conv_kernel_size' parameter (similarily, 'conv_kernel_size'
      should be set to -1 if 'from_layer' is specified). The created convolution
      operation will be a normal 2D convolution by default, and a depthwise
      convolution followed by 1x1 convolution if 'use_depthwise' is set to True.
    depth_multiplier: Depth multiplier for convolutional layers.
    min_depth: Minimum depth for convolutional layers.
    insert_1x1_conv: A boolean indicating whether an additional 1x1 convolution
      should be inserted before shrinking the feature map.
    image_features: A dictionary of handles to activation tensors from the
      base feature extractor.

  Returns:
    feature_maps: an OrderedDict mapping keys (feature map names) to
      tensors where each tensor has shape [batch, height_i, width_i, depth_i].

  Raises:
    ValueError: if the number entries in 'from_layer' and
      'layer_depth' do not match.
    ValueError: if the generated layer does not have the same resolution
      as specified.
  """
  depth_fn = get_depth_fn(depth_multiplier, min_depth)

  feature_map_keys = []
  feature_maps = []
  base_from_layer = ''
  use_explicit_padding = False
  if 'use_explicit_padding' in feature_map_layout:
    use_explicit_padding = feature_map_layout['use_explicit_padding']
  use_depthwise = False
  if 'use_depthwise' in feature_map_layout:
    use_depthwise = feature_map_layout['use_depthwise']
  for index, from_layer in enumerate(feature_map_layout['from_layer']):
    layer_depth = feature_map_layout['layer_depth'][index]
    conv_kernel_size = 3
    if 'conv_kernel_size' in feature_map_layout:
      conv_kernel_size = feature_map_layout['conv_kernel_size'][index]
    if from_layer:
      feature_map = image_features[from_layer]
      base_from_layer = from_layer
      feature_map_keys.append(from_layer)
    else:
      pre_layer = feature_maps[-1]
      intermediate_layer = pre_layer
      if insert_1x1_conv:
        layer_name = '{}_1_Conv2d_{}_1x1_{}'.format(
            base_from_layer, index, depth_fn(layer_depth / 2))
        intermediate_layer = slim.conv2d(
            pre_layer,
            depth_fn(layer_depth / 2), [1, 1],
            padding='SAME',
            stride=1,
            scope=layer_name)
      layer_name = '{}_2_Conv2d_{}_{}x{}_s2_{}'.format(
          base_from_layer, index, conv_kernel_size, conv_kernel_size,
          depth_fn(layer_depth))
      stride = 2
      padding = 'SAME'
      if use_explicit_padding:
        padding = 'VALID'
        intermediate_layer = ops.fixed_padding(
            intermediate_layer, conv_kernel_size)
      if use_depthwise:
        feature_map = slim.separable_conv2d(
            intermediate_layer,
            None, [conv_kernel_size, conv_kernel_size],
            depth_multiplier=1,
            padding=padding,
            stride=stride,
            scope=layer_name + '_depthwise')
        feature_map = slim.conv2d(
            feature_map,
            depth_fn(layer_depth), [1, 1],
            padding='SAME',
            stride=1,
            scope=layer_name)
      else:
        feature_map = slim.conv2d(
            intermediate_layer,
            depth_fn(layer_depth), [conv_kernel_size, conv_kernel_size],
            padding=padding,
            stride=stride,
            scope=layer_name)
      feature_map_keys.append(layer_name)
    feature_maps.append(feature_map)
  return collections.OrderedDict(
      [(x, y) for (x, y) in zip(feature_map_keys, feature_maps)])





