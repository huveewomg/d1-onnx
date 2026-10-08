import numpy as np, onnx, onnxruntime as ort, sys
from onnx import helper, TensorProto
# build one-op graph per op type from argv
which = sys.argv[1]
x = np.random.default_rng(0).standard_normal((1,1024,768)).astype(np.float32)
inits = []
if which == 'LayerNormalization':
    scale = helper.make_tensor('scale', TensorProto.FLOAT, [768], np.ones(768, dtype=np.float32).tolist())
    bias = helper.make_tensor('bias', TensorProto.FLOAT, [768], np.zeros(768, dtype=np.float32).tolist())
    inits = [scale, bias]
    node = helper.make_node('LayerNormalization', ['X','scale','bias'], ['Y'], axis=-1, name='ln')
    g_in = [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1,1024,768])]
    g_out = [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1,1024,768])]
elif which == 'Tanh':
    node = helper.make_node('Tanh', ['X'], ['Y'], name='tn')
    g_in = [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1,1024,768])]
    g_out = [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1,1024,768])]
elif which == 'IsNaN':
    node = helper.make_node('IsNaN', ['X'], ['Y'], name='isnan')
    g_in = [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1,1024,768])]
    g_out = [helper.make_tensor_value_info('Y', TensorProto.BOOL, [1,1024,768])]
elif which == 'Gelu_via_Tanh':
    n1 = helper.make_node('Mul', ['X','X'], ['X2'], name='mul')
    half = helper.make_tensor('half', TensorProto.FLOAT, [], [0.044715])
    three = helper.make_tensor('three', TensorProto.FLOAT, [], [0.7978845608028654])
    inits = [half, three]
    n2 = helper.make_node('Mul', ['X2','half'], ['X2b'], name='mul2')
    n3 = helper.make_node('Add', ['X','X2b'], ['X3'], name='add')
    n4 = helper.make_node('Mul', ['X3','three'], ['X3b'], name='mul3')
    n5 = helper.make_node('Tanh', ['X3b'], ['T'], name='tanh')
    n6 = helper.make_node('Add', ['T','one'], ['T1'], name='add2')
    one = helper.make_tensor('one', TensorProto.FLOAT, [], [1.0])
    inits.append(one)
    n7 = helper.make_node('Mul', ['X','T1'], ['Y'], name='mul4')
    g = [n1,n2,n3,n4,n5,n6,n7]
    g_in = [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1,1024,768])]
    g_out = [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1,1024,768])]
graph = helper.make_graph(g if which=='Gelu_via_Tanh' else [node], f'probe_{which}', g_in, g_out, inits)
m = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)])
m.ir_version = 9
onnx.checker.check_model(m)
onnx.save(m, f'probe_{which}.onnx')
s = ort.InferenceSession(f'probe_{which}.onnx', providers=['CPUExecutionProvider'])
o = s.run(None, {'X': x})[0]
print('cpu ok', o.shape)
