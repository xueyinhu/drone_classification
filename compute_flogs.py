# https://zhuanlan.zhihu.com/p/576586123

import tensorflow as tf

from typing import Any, Callable, Dict, List, Optional, Union
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

from config import get_config
from nets import *


def try_count_flops(model: Union[tf.Module, tf.keras.Model], inputs_kwargs: Optional[Dict[str, Any]] = None, output_path: Optional[str] = None):
    if hasattr(model, 'inputs'):
        try:
            # Get input shape and set batch size to 1.
            if model.inputs:
                inputs = [
                    tf.TensorSpec([1] + input.shape[1:], input.dtype)
                    for input in model.inputs
                ]
                concrete_func = tf.function(model).get_concrete_function(inputs)
            # If model.inputs is invalid, try to use the input to get concrete
            # function for model.call (subclass model).
            else:
                concrete_func = tf.function(model.call).get_concrete_function(
                    **inputs_kwargs)
            frozen_func, _ = convert_variables_to_constants_v2_as_graph(concrete_func)

            # Calculate FLOPs.
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            if output_path is not None:
                opts['output'] = f'file:outfile={output_path}'
            else:
                opts['output'] = 'none'
            flops = tf.compat.v1.profiler.profile(
                graph=frozen_func.graph, run_meta=run_meta, options=opts)
            return flops.total_float_ops
        except Exception as e:  # pylint: disable=broad-except
            logging.info(
                'Failed to count model FLOPs with error %s, because the build() '
                'methods in keras layers were not called. This is probably because '
                'the model was not feed any input, e.g., the max train step already '
                'reached before this run.', e)
            return None
    return None

i = tf.keras.Input((960, 320, 3))
m = tf.keras.Model(i, MyNet(get_config())(i))
flops = try_count_flops(m)
print(flops/1000000,"M Flops")