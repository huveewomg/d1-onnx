"""Validate a graph's external-data references before packing/uploading.

Walks BOTH surfaces where external data can hide:
  - graph.initializer TensorProtos
  - node-attribute tensors (Constant nodes etc.) — the surface the int8
    external-data bug (HF fa3184e fix) came from: the pack/merge step rewrote
    initializer locations but was blind to node attributes, and a fresh
    download failed ORT external-data validation.

Passes a directory of local files (--files) or an HF repo file list
(--repo; read-only, no token needed for public repos).

Exit 0 = every referenced location exists; exit 1 + listing = missing refs.
Usage:
  python validate_external_refs.py <graph.onnx> --files <dir>
  python validate_external_refs.py <graph.onnx> --repo NAMESPACE/NAME[/REVISION]
                                   [--graph-path <path-in-repo>]
"""
import argparse
import os
import posixpath

import onnx
from onnx import AttributeProto, TensorProto


def collect_locations(model):
    """(surface, tensor_name, location) for every external-data reference."""
    refs = []

    def scan_tensor(t, surface, where):
        if t.data_location == TensorProto.EXTERNAL:
            loc = next((kv.value for kv in t.external_data if kv.key == "location"), None)
            refs.append((surface, where, loc))

    for i in model.graph.initializer:
        scan_tensor(i, "initializer", i.name)
    for n in model.graph.node:
        for a in n.attribute:
            if a.type == AttributeProto.TENSOR:
                scan_tensor(a.t, "node-attr", f"{n.name}:{a.t.name}")
            elif a.type == AttributeProto.TENSORS:
                for tt in a.tensors:
                    scan_tensor(tt, "node-attr", f"{n.name}:{tt.name}")
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("graph")
    ap.add_argument("--files", help="directory holding the external data files (local mode)")
    ap.add_argument("--repo", metavar="NAMESPACE/NAME[/REVISION]", help="HF repo file list (remote mode)")
    ap.add_argument("--graph-path", default=None,
                    help="the graph's path INSIDE the repo (remote mode; default: basename of the local graph file)")
    args = ap.parse_args()

    if bool(args.files) == bool(args.repo):
        ap.error("exactly one of --files or --repo required")

    model = onnx.load(args.graph, load_external_data=False)
    refs = collect_locations(model)
    if not refs:
        print("no external-data references — nothing to validate")
        return

    if args.repo:
        from huggingface_hub import HfApi
        parts = args.repo.split("/")
        if len(parts) == 2:            # NAMESPACE/NAME -> main
            repo_id, rev = args.repo, "main"
        elif len(parts) == 3:          # NAMESPACE/NAME/REVISION
            repo_id, rev = "/".join(parts[:2]), parts[2]
        else:
            ap.error("--repo must be NAMESPACE/NAME[/REVISION]")
        avail = {s.rfilename for s in HfApi().model_info(repo_id, revision=rev, files_metadata=True).siblings}
        in_repo = (args.graph_path or os.path.basename(args.graph)).replace(os.sep, "/")
        base_dir = posixpath.dirname(in_repo)

        def exists(loc):
            return (posixpath.join(base_dir, loc) if base_dir else loc) in avail
    else:
        def exists(loc):
            return os.path.exists(os.path.join(args.files, loc))

    bad = [r for r in refs if r[2] is None]
    locations = sorted({loc for _, _, loc in refs if loc is not None})
    missing = [loc for loc in locations if not exists(loc)]
    surfaces = sorted({s for s, _, _ in refs})
    print(f"{len(refs)} external-data references across surfaces {surfaces}")
    print(f"distinct locations: {len(locations)}; missing: {len(missing)}; null-location refs: {len(bad)}")
    if bad:
        for s, w, _ in bad[:5]:
            print(f"  MISSING location key: [{s}] {w}")
    if missing:
        for loc in missing:
            n_ = sum(1 for s, _, l_ in refs if l_ == loc)
            print(f"  MISSING file: {loc!r}  ({n_} tensors reference it)")
        raise SystemExit(1)
    if bad:
        raise SystemExit(1)
    print("all referenced external files present")


if __name__ == "__main__":
    main()