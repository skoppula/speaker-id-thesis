from helpers.helpers import get_tensors_from_graph
from tensorflow.contrib import graph_editor as ge
import tensorflow as tf

_ = tf.train.import_meta_graph("/data/sls/u/meng/skanda/home/thesis/manfxpt/models/sentfiltNone_fcn2_bnTrue_regFalse_noLRSchedule/graph-0215-163806.meta")
g = tf.get_default_graph()
co = ge.ControlOutputs(g)
relu_ops = []
for op in g.get_operations():
    if 'Relu' in op.type and 'tower0' in op.name and 'gradients' not in op.name:
        relu_ops.append(op)
        print(op.name, op.type, op.values())
        print(co.get(op))
        # print(op.control_inputs, ge.get_ops_ios(op, False))
    if 'wrapper' in op.name:
        print(op.name, op.type, op.values())
        # print(op.control_inputs, ge.get_ops_ios(op, False))

for i, relu_op in enumerate(relu_ops):
    relu_node = g.get_tensor_by_name(relu_op.name + ":0")
    new_name = '/'.join(relu_op.name.split('/')[:-1]) + "/wrapper_to_relu_" + str(i)
    wrapper = tf.identity(relu_node, name=new_name)
    # ge.detach_inputs(wrapper)
    # ge.swap_outputs(relu_node, wrapper)
    # ge.connect(relu_node, wrapper)

print("here", relu_node, wrapper)
post_relu_ops = ge.get_consuming_ops([relu_node])
print(post_relu_ops, ge.get_consuming_ops([wrapper]))
wrapper_op = tf.get_default_graph().get_operation_by_name("tower0/linear3/output")
print(wrapper_op, post_relu_ops[0])
ge.connect(wrapper_op, post_relu_ops[0])
print(post_relu_ops, ge.get_consuming_ops([wrapper]))
    
print("after")
for op in g.get_operations():
    if 'wrapper' in op.name:
        print(op.name, op.type, op.values())
        # node = g.get_tensor_by_name(op.name + ":0")
        # print(ge.sgv(node))
    if 'Relu' in op.type and 'tower0' in op.name and 'gradients' not in op.name:
        print(op.name, op.type, op.values())
        # node = g.get_tensor_by_name(op.name + ":0")
        # print(ge.sgv(node))
        
# saver = tf.train.Saver()
a = tf.train.export_meta_graph('wrapperrelutest/model.meta', as_text=True)
writer = tf.summary.FileWriter('wrapperrelutest/', tf.get_default_graph())
writer.close()
