import numpy as np, onnx, onnxruntime as ort, sys
from onnx import helper, TensorProto
which = sys.argv[1]
x = np.random.default_rng(0).standard_normal((1,1024,768)).astype(np.float32)
if which == 'IsNaN_Where':
    # pattern from NaFlex: Where(IsNaN(x), zero, x)
    zero = helper.make_tensor('zero', TensorProto.FLOAT, [1], [0.0])
    node = helper.make_node('IsNaN', ['X'], ['N'], name='isnan')
    node2 = helper.make_node('Where', ['N','zero','X'], ['Y'], name='where')
    g = [node, node2]
    g_in = [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1,1024,768])]
    g_out = [helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1,1024,768])]
    inits=[zero]
elif which == 'Where_int32':
    # Where on int32 mask + int64 shapes (aux input pattern)
    cond = helper.make_tensor_value_info('C', TensorProto.BOOL, [1,1024])
    A = helper.make_tensor_value_info('A', TensorProto.FLOAT, [1,1024,768])
    node = helper.make_node('Where', ['C','A','A2'], ['Y'], name='where')
    zero = helper.make_tensor('A2', TensorProto.FLOAT, [1], [0.0])
    inits=[zero]
    g_in=[cond, A]; g_out=[helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1,1024,768])]
    g=[node]
elif which == 'PatchGather':
    # gather on positions (positional embedding interpolation path)
    table = np.random.default_rng(1).standard_normal((1030, 1152)).astype(np.float32)
    t = helper.make_tensor('T', TensorProto.FLOAT, table.shape, table.flatten().tolist())
    idx = helper.make_tensor_value_info('IDX', TensorProto.INT64, [1,1024])
    node = helper.make_node('Gather', ['T','IDX'], ['Y'], axis=0, name='g')
    inits=[t]; g_in=[idx]; g_out=[helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1,1024,1152])]; g=[node]
graph = helper.make_graph(g, f'probe_{which}', g_in, g_out, inits)
m = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 18)]); m.ir_version = 9
onnx.checker.check_model(m)
onnx.save(m, f'probe_{which}.onnx')
s = ort.InferenceSession(f'probe_{which}.onnx', providers=['CPUExecutionProvider'])
if which == 'IsNaN_Where': o = s.run(None, {'X': x})[0]
elif which == 'Where_int32': o = s.run(None, {'C': np.ones((1,1024),bool), 'A': np.zeros((1,1024,768),np.float32)+np.arange(768,dtype=np.float32)})[0]
else: o = s.run(None, {'IDX': np.arange(1024, dtype=np.int64)[None,:]})[0]
print('cpu ok', o.shape, o.dtype)
