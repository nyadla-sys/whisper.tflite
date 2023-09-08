# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Test configs for sigmoid grad op."""
import tensorflow.compat.v1 as tf
from tensorflow.lite.testing.zip_test_utils import create_tensor_data
from tensorflow.lite.testing.zip_test_utils import make_zip_of_tests
from tensorflow.lite.testing.zip_test_utils import register_make_test_function


@register_make_test_function()
def make_sigmoid_grad_tests(options):
  """Make a set of tests to examine sigmoid grad op."""

  test_parameters = [{
      "dtype": [tf.float32],
      "input_shape": [[3, 2, 1], [1, 2, 3, 4]],
      "input_range": [(-32, 32)],
  }]

  def build_graph(parameters):
    y = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="y", shape=parameters["input_shape"])
    dy = tf.compat.v1.placeholder(
        dtype=parameters["dtype"], name="dy", shape=parameters["input_shape"])
    out = tf.raw_ops.SigmoidGrad(y=y, dy=dy)
    return [y, dy], [out]

  def build_inputs(parameters, sess, inputs, outputs):
    # input is y which is sigmoid(x), hence it's range is between 0 to 1
    y = create_tensor_data(
        parameters["dtype"],
        parameters["input_shape"],
        min_value=0,
        max_value=1)
    min_value, max_value = parameters["input_range"]
    dy = create_tensor_data(parameters["dtype"], parameters["input_shape"],
                            min_value, max_value)
    values = [y, dy]
    return values, sess.run(outputs, feed_dict=dict(zip(inputs, values)))

  make_zip_of_tests(options, test_parameters, build_graph, build_inputs)
