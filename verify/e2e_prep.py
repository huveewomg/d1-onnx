"""E2E prep (venv): build image-decision inputs with LIQUID'S OWN GLUE + torch references.
Saves e2e_inputs.npz (+ ref probs) for the NPU stage (ryzen env has no transformers 5.x).
"""
import io, os, sys
import numpy as np
import torch
from PIL import Image
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "models", "d1-3B"))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
sys.path.insert(0, MODEL_DIR)

from transformers import AutoModel, Lfm2VlForConditionalGeneration
import prompt as d1_prompt

CATS_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
QUESTIONS = {"cats": {"type": "choice", "instructions": "How many cats are there?",
                      "criteria": {"one": "One", "two": "Two", "more": "Three or more"}}}

# liquid's own engine (relative imports already wired by trust_remote_code)
remote = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, dtype=torch.float32).eval()
engine = remote.engine
tok = engine.tokenizer

im512 = Image.open(io.BytesIO(urllib.request.urlopen(CATS_URL).read())).convert("RGB").resize((512, 512))
q = d1_prompt.as_question(QUESTIONS["cats"])

# their canonical image-input builder: markup + processor on full prompt
markup = engine._image_markup(1)
print("markup:", [markup])
prefix = d1_prompt.prefix_text(tok, None, "", style="json_only", system="none", images=markup)
suffix = d1_prompt.suffix_text(tok, q, "", "desc")
inputs = engine._image_inputs(prefix + suffix, [im512])
ids = inputs["input_ids"]
pv, pam, ss = inputs["pixel_values"], inputs["pixel_attention_mask"], inputs["spatial_shapes"]
IMG_TOKEN = 124907
print("ids:", tuple(ids.shape), "img tokens:", int((ids[0] == IMG_TOKEN).sum()),
      "pixels:", tuple(pv.shape), "patch sums:", pam.sum(1).tolist(), "spatial:", ss.tolist())

# torch native reference: SAME inputs through the native language model (no engine glue)
m = Lfm2VlForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.float32).eval()
for p in m.parameters(): p.requires_grad_(False)
with torch.no_grad():
    # top-level forward: the image merge happens in Lfm2VlModel.forward
    hidden = m.model(input_ids=ids, pixel_values=pv, spatial_shapes=ss,
                     pixel_attention_mask=pam, use_cache=False).last_hidden_state
ref_logits = m.lm_head(hidden[:, -1, :])[0].numpy()
ref_probs = np.array(d1_prompt.readout(tok, q, ref_logits - ref_logits.max()))
print("torch-native (same inputs) probs:", [f"{v:.4f}" for v in ref_probs])

# THEIR semantic ground truth through the whole engine
with torch.inference_mode():
    res = remote.system_one(None, QUESTIONS, images=[im512])
print("system_one:", res["answers"]["cats"])

# host-merge needs: positions + option tokens + image token
out_path = os.path.join(HERE, "e2e_inputs.npz")
np.savez(out_path,
         input_ids=ids.numpy(),
         pixel_values=pv.numpy().astype(np.float32),
         pixel_attention_mask=pam.numpy().astype(np.int32),
         spatial_shapes=ss.numpy(),
         image_positions=(ids[0] == IMG_TOKEN).nonzero(as_tuple=True)[0].numpy().astype(np.int64),
         option_token_ids=np.array([t for g in d1_prompt.readout_ids(tok, q) for t in g], dtype=np.int64),
         ref_probs=ref_probs)
print("saved", out_path)
