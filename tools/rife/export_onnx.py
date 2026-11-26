#!/usr/bin/env python3
"""RIFE v4.22-lite / v4.25-lite -> ONNX for moonglide's TensorRT frame generation (see README.md)."""
import argparse
import os
import sys
import urllib.request
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import importlib  # noqa: E402

URL = "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v{v}.lite.pkl"
MODELS = ("4.22-lite", "4.25-lite")


class Rife(torch.nn.Module):
    """img0, img1 (1,3,H,W) 0..1 and t (1,1,1,1) -> the frame at t."""

    def __init__(self, net, w, h, align=32):
        super().__init__()
        self.net = net
        # 4.25 runs a 1/32 block, so it needs 128-aligned sizes: pad inside the graph, crop on output
        self.h0, self.w0 = h, w
        w, h = -(-w // align) * align, -(-h // align) * align
        gx = torch.linspace(-1.0, 1.0, w).view(1, 1, 1, w).expand(1, 1, h, w)
        gy = torch.linspace(-1.0, 1.0, h).view(1, 1, h, 1).expand(1, 1, h, w)
        self.register_buffer("grid", torch.cat([gx, gy], 1).contiguous())
        self.register_buffer("div", torch.tensor([(w - 1.0) / 2.0, (h - 1.0) / 2.0]))
        self.h, self.w = h, w

    def forward(self, img0, img1, t):
        pad = (0, self.w - self.w0, 0, self.h - self.h0)
        if pad[1] or pad[3]:
            img0 = torch.nn.functional.pad(img0, pad, mode="replicate")
            img1 = torch.nn.functional.pad(img1, pad, mode="replicate")
        f0 = self.net.encode(img0)
        f1 = self.net.encode(img1)
        ts = t.expand(1, 1, self.h, self.w)
        out = self.net(img0, img1, ts, self.div, self.grid, f0, f1)
        return out[:, :, : self.h0, : self.w0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="4.22-lite", choices=MODELS)
    ap.add_argument("--size", default="1920x1088", help="WxH, multiples of 32")
    ap.add_argument("--dir", default=os.path.expanduser("~/.local/share/moonglide/rife"))
    a = ap.parse_args()
    ver = a.model.split("-")[0]
    rife = importlib.import_module("IFNet_HDv3_v" + ver.replace(".", "_") + "_lite")
    url = URL.format(v=ver)
    w, h = (int(v) for v in a.size.split("x"))
    if w % 32 or h % 32:
        sys.exit("size must be a multiple of 32 (pad the picture)")
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    weights = d / f"flownet_v{ver}.lite.pkl"
    if not weights.exists():
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, weights.with_suffix(".part"))
        weights.with_suffix(".part").replace(weights)
    state = torch.load(weights, map_location="cpu", weights_only=True)
    state = {k.replace("module.", ""): v for k, v in state.items() if k.startswith("module.")}
    net = rife.IFNet()
    missing = net.load_state_dict(state, strict=False).missing_keys
    if missing:
        sys.exit(f"checkpoint is missing {missing[:5]}")
    model = Rife(net.eval(), w, h, 128 if ver == "4.25" else 32).eval()
    out = d / f"rife_v{ver}_lite_{w}x{h}.onnx"
    x = torch.rand(1, 3, h, w)
    with torch.no_grad():
        torch.onnx.export(model, (x, x.clone(), torch.full((1, 1, 1, 1), 0.5)), str(out), opset_version=17,
                          input_names=["img0", "img1", "t"], output_names=["mid"], dynamo=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
