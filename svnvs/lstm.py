"""Module for constructing RNN Cells."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

from tensorflow.contrib.compiler import jit
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import initializers
from tensorflow.python.keras.engine import input_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_impl  # pylint: disable=unused-import
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import partitioned_variables  # pylint: disable=unused-import
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import nest
import tensorflow as tf
class ConvLSTMCell(rnn_cell_impl.RNNCell):
  """Convolutional LSTM recurrent network cell.
  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self,
               conv_ndims,
               input_shape,
               output_channels,
               kernel_shape,
               dilation=1,
               use_bias=True,
               skip_connection=False,
               forget_bias=1.0,
               initializers=None,
               name="conv_lstm_cell"):
    """Construct ConvLSTMCell.
    Args:
      conv_ndims: Convolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as int tuple, excluding the batch size.
      output_channels: int, number of output channels of the conv LSTM.
      kernel_shape: Shape of kernel as an int tuple (of size 1, 2 or 3).
      use_bias: (bool) Use bias in convolutions.
      skip_connection: If set to `True`, concatenate the input to the
        output of the conv LSTM. Default: `False`.
      forget_bias: Forget bias.
      initializers: Unused.
      name: Name of the module.
    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(ConvLSTMCell, self).__init__(name=name)

    if conv_ndims != len(input_shape) - 1:
      raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
          input_shape, conv_ndims))
    self._dilation=dilation
    self._conv_ndims = conv_ndims
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_shape = list(kernel_shape)
    self._use_bias = use_bias
    self._forget_bias = forget_bias
    self._skip_connection = skip_connection

    self._total_output_channels = output_channels
    if self._skip_connection:
      self._total_output_channels += self._input_shape[-1]

    state_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._output_channels])
    self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
    self._output_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._total_output_channels])

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state,scope=None):
    cell, hidden = state
    # with vs.variable_scope(scope, reuse=tf.AUTO_REUSE):
    new_hidden = _conv([inputs, hidden], self._kernel_shape,
                      4 * self._output_channels, self._use_bias,dilations=[1,1,1,1],name="kernel")


    gates = array_ops.split(
        value=new_hidden, num_or_size_splits=4, axis=self._conv_ndims + 1)

    input_gate, new_input, forget_gate, output_gate = gates
    new_cell = math_ops.sigmoid(forget_gate + self._forget_bias) * cell
    new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)
    output = math_ops.tanh(new_cell) * math_ops.sigmoid(output_gate)

    if self._skip_connection:
      output = array_ops.concat([output, inputs], axis=-1)
    new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
    return output, new_state
def _conv(args, filter_size, num_features, bias, bias_start=0.0,dilations=[1,1,1,1],name="kernel"):
  """Convolution.
  Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
    batch x n, Tensors.
    filter_size: int tuple of filter shape (of size 1, 2 or 3).
    num_features: int, number of features.
    bias: Whether to use biases in the convolution layer.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  shape_length = len(shapes[0])
  for shape in shapes:
    if len(shape) not in [3, 4, 5]:
      raise ValueError("Conv Linear expects 3D, 4D "
                       "or 5D arguments: %s" % str(shapes))
    if len(shape) != len(shapes[0]):
      raise ValueError("Conv Linear expects all args "
                       "to be of same Dimension: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[-1]
  dtype = [a.dtype for a in args][0]

  # determine correct conv operation
  if shape_length == 3:
    conv_op = nn_ops.conv1d
    strides = 1
  elif shape_length == 4:
    conv_op = nn_ops.conv2d
    strides = shape_length * [1]
  elif shape_length == 5:
    conv_op = nn_ops.conv3d
    strides = shape_length * [1]

  # Now the computation.
  kernel = vs.get_variable(
      name, filter_size + [total_arg_size_depth, num_features], dtype=dtype)
  if len(args) == 1:
    res = conv_op(args[0], kernel, strides,dilations=dilations, padding="SAME")
  else:
    res = conv_op(
        array_ops.concat(axis=shape_length - 1, values=args),
        kernel,
        strides,
        dilations=dilations,
        padding="SAME")
  if not bias:
    return res
  bias_term = vs.get_variable(
      "biases", [num_features],
      dtype=dtype,
      initializer=init_ops.constant_initializer(bias_start, dtype=dtype))
  return res + bias_term

def _deconv(args, filter_size, num_features, bias, bias_start=0.0,dilations=1,name="kernel"):
  """Convolution.
  Args:
    args: a Tensor or a list of Tensors of dimension 3D, 4D or 5D,
    batch x n, Tensors.
    filter_size: int tuple of filter shape (of size 1, 2 or 3).
    num_features: int, number of features.
    bias: Whether to use biases in the convolution layer.
    bias_start: starting value to initialize the bias; 0 by default.
  Returns:
    A 3D, 4D, or 5D Tensor with shape [batch ... num_features]
  Raises:
    ValueError: if some of the arguments has unspecified or wrong shape.
  """

  # Calculate the total size of arguments on dimension 1.
  total_arg_size_depth = 0
  shapes = [a.get_shape().as_list() for a in args]
  shape_length = len(shapes[0])
  for shape in shapes:
    if len(shape) not in [3, 4, 5]:
      raise ValueError("Conv Linear expects 3D, 4D "
                       "or 5D arguments: %s" % str(shapes))
    if len(shape) != len(shapes[0]):
      raise ValueError("Conv Linear expects all args "
                       "to be of same Dimension: %s" % str(shapes))
    else:
      total_arg_size_depth += shape[-1]
  dtype = [a.dtype for a in args][0]

  # determine correct conv operation
  if shape_length == 3:
    conv_op = nn_ops.conv1d_transpose
    strides = 1
  elif shape_length == 4:
    conv_op = nn_ops.conv2d_transpose
    strides = shape_length * [1]
  elif shape_length == 5:
    conv_op = nn_ops.conv3d_transpose
    strides = shape_length * [1]

  # Now the computation.
  kernel = vs.get_variable(
      name, filter_size + [total_arg_size_depth, num_features], dtype=dtype)
  if len(args) == 1:
    res = conv_op(args[0], kernel, strides,dilations=dilations, padding="SAME")
  else:
    res = conv_op(
        array_ops.concat(axis=shape_length - 1, values=args),
        kernel,
        strides,
        dilations=dilations,
        padding="SAME")
  if  bias:

    res = vs.get_variable(
        "biases", [num_features],
        dtype=dtype,
        initializer=init_ops.constant_initializer(bias_start, dtype=dtype))

  return res

_bn=tf.layers.batch_normalization
class ConvBnLSTMCell(ConvLSTMCell):
  def __init__(self, conv_ndims, input_shape, output_channels, kernel_shape, dilation=1, use_bias=True, skip_connection=False, forget_bias=1.0, initializers=None, name='conv_lstm_cell'):
    super(ConvBnLSTMCell, self).__init__(conv_ndims, input_shape, output_channels, kernel_shape, dilation=dilation, use_bias=use_bias, skip_connection=skip_connection, forget_bias=forget_bias, initializers=initializers, name=name)
    self._conv=_conv_bn
class DeConvBnLSTMCell(ConvLSTMCell):
  def __init__(self, conv_ndims, input_shape, output_channels, kernel_shape, dilation=1, use_bias=True, skip_connection=False, forget_bias=1.0, initializers=None, name='conv_lstm_cell'):
    super(ConvBnLSTMCell, self).__init__(conv_ndims, input_shape, output_channels, kernel_shape, dilation=dilation, use_bias=use_bias, skip_connection=skip_connection, forget_bias=forget_bias, initializers=initializers, name=name)
    self._conv=_deconv_bn
def _deconv_bn(args, filter_size, num_features, bias, bias_start=0.0,dilations=1,relu=False,name="kernel"):
  res=_deconv(args, filter_size, num_features, bias, bias_start=0.0,dilations=1,name="kernel")
  res=_bn(res,training=True,reuse=tf.AUTO_REUSE,name=name+"_bn")
  if relu:
    res=tf.nn.relu(res)
  return res
def _conv_bn(args, filter_size, num_features, bias, bias_start=0.0,dilations=1,relu=False,name="kernel"):
  res=_conv(args, filter_size, num_features, bias, bias_start=bias_start,dilations=dilations,name=name)
  res=_bn(res,training=True,reuse=tf.AUTO_REUSE,name=name+"_bn")
  if relu:
    res=tf.nn.relu(res)
  return res



class ConvsLSTMCell(rnn_cell_impl.RNNCell):
  """Convolutional LSTM recurrent network cell.
  https://arxiv.org/pdf/1506.04214v1.pdf
  """

  def __init__(self,
               conv_ndims,
               input_shape,
               output_channels,
               kernel_shape,
               dilation=1,
               use_bias=True,
               skip_connection=False,
               forget_bias=1.0,
               initializers=None,
               name="conv_lstm_cell"):
    """Construct ConvLSTMCell.
    Args:
      conv_ndims: Convolution dimensionality (1, 2 or 3).
      input_shape: Shape of the input as int tuple, excluding the batch size.
      output_channels: int, number of output channels of the conv LSTM.
      kernel_shape: Shape of kernel as an int tuple (of size 1, 2 or 3).
      use_bias: (bool) Use bias in convolutions.
      skip_connection: If set to `True`, concatenate the input to the
        output of the conv LSTM. Default: `False`.
      forget_bias: Forget bias.
      initializers: Unused.
      name: Name of the module.
    Raises:
      ValueError: If `skip_connection` is `True` and stride is different from 1
        or if `input_shape` is incompatible with `conv_ndims`.
    """
    super(ConvsLSTMCell, self).__init__(name=name)

    if conv_ndims != len(input_shape) - 1:
      raise ValueError("Invalid input_shape {} for conv_ndims={}.".format(
          input_shape, conv_ndims))
    self._dilation=dilation
    self._conv_ndims = conv_ndims
    self._input_shape = input_shape
    self._output_channels = output_channels
    self._kernel_shape = list(kernel_shape)
    self._use_bias = use_bias
    self._forget_bias = forget_bias
    self._skip_connection = skip_connection

    self._total_output_channels = output_channels
    if self._skip_connection:
      self._total_output_channels += self._input_shape[-1]

    state_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._output_channels])
    self._state_size = rnn_cell_impl.LSTMStateTuple(state_size, state_size)
    self._output_size = tensor_shape.TensorShape(
        self._input_shape[:-1] + [self._total_output_channels])

  @property
  def output_size(self):
    return self._output_size

  @property
  def state_size(self):
    return self._state_size

  def call(self, inputs, state,scope=None):
    cell, hidden = state
    # with vs.variable_scope(scope, reuse=tf.AUTO_REUSE):
    new_hidden = _conv_bn([inputs, hidden], self._kernel_shape,
                      1 * self._output_channels,False,dilations=1,relu=True,name="kernel0")
    new_hidden = _conv_bn([new_hidden], self._kernel_shape,
                      2 * self._output_channels, False,dilations=1,relu=True,name="kernel1")
    new_hidden = _conv([new_hidden], self._kernel_shape,
                      4 * self._output_channels, self._use_bias,dilations=1,name="kernel2")


    gates = array_ops.split(
        value=new_hidden, num_or_size_splits=4, axis=self._conv_ndims + 1)

    input_gate, new_input, forget_gate, output_gate = gates
    new_cell = math_ops.sigmoid(forget_gate + self._forget_bias) * cell
    new_cell += math_ops.sigmoid(input_gate) * math_ops.tanh(new_input)
    output = math_ops.tanh(new_cell) * math_ops.sigmoid(output_gate)

    if self._skip_connection:
      output = array_ops.concat([output, inputs], axis=-1)
    new_state = rnn_cell_impl.LSTMStateTuple(new_cell, output)
    return output, new_state