import os, math
import numpy as np, onnxruntime as ort
import sys
MODEL_DIR = os.path.abspath('../../models/d1-3B')
os.environ.setdefault('HF_HUB_OFFLINE','1'); sys.path.insert(0, MODEL_DIR)
from transformers import AutoTokenizer
import prompt as d1_prompt
tok = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
q = d1_prompt.as_question({"type": "choice", "instructions": "How many cats are there?",
                           "criteria": {"one": "One", "two": "Two", "more": "Three or more"}})
labels = ['one','two','more']
d = np.load('e2e_inputs.npz'); ids = d['input_ids'][0]; img_pos = d['image_positions']
groups = np.array([g for g in d1_prompt.readout_ids(tok, q)])  # (3, <=2)
ref_probs = d['ref_probs']
table = np.memmap(os.path.join(HERE,'export','lm.embed_tokens.weight'), dtype='<f4', mode='r').reshape(128000, 2048) if (HERE:=os.path.dirname(os.path.abspath(__file__))) else None

def merged(img_embeds):
    e = table[ids].copy(); e[img_pos] = img_embeds[:]; return e

def run_text(em):
    L = len(ids); pad = 512 - L
    em5 = np.zeros((1,512,2048), dtype=np.float32); em5[0,pad:] = em
    mask = np.zeros((1,512), dtype=np.int64); mask[0,pad:] = 1
    pos = np.clip(np.cumsum(mask,axis=1)-1, 0, None).astype(np.int64)
    s = ort.InferenceSession('export/d1-3B_embeds_pass_seq512.onnx', providers=['CPUExecutionProvider'])
    return s.run(None, {'inputs_embeds': em5, 'attention_mask': mask, 'position_ids': pos})[0][0,0]

def readout(lz):
    scores = [max(float(lz[i]) for i in g) for g in groups]
    mx = max(scores); ex = [math.exp(s-mx) for s in scores]
    return [e/sum(ex) for e in ex]

img_cpu = ort.InferenceSession('export/d1-3B_vision_tile512.onnx', providers=['CPUExecutionProvider']
    ).run(None, {'pixel_values': d['pixel_values']})[0]
img_npu = np.load('e2e_img_embeds_npu.npz')['embeds']

p_cpu = readout(run_text(merged(img_cpu)))
p_npu = readout(run_text(merged(img_npu)))
print(f"torch-native (same inputs) : [one/two/more] {[round(float(v),4) for v in ref_probs]}  -> {labels[int(np.argmax(ref_probs))]}")
print(f"ONNX + CPU vision          : {[round(v,4) for v in p_cpu]}  -> {labels[int(np.argmax(p_cpu))]}")
print(f"ONNX + NPU  vision         : {[round(v,4) for v in p_npu]}  -> {labels[int(np.argmax(p_npu))]}")
print(f"NPU prob drift vs torch    : {float(np.max(np.abs(np.array(p_npu)-ref_probs))):.4f}")
print("E2E_PASS" if np.argmax(p_npu)==np.argmax(ref_probs) else "E2E_PROB_DRIFT_BUT_CHECK")
