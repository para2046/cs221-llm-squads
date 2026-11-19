"""Export each tab of artifacts/battle_replay/demo3_video/index.html as a 1920x1080 presentation mp4:
the SC2 rendering (with a0-a4 / e0-e4 labels) on the left, step / orders / team chat / kill feed on the right.
1 video frame per game loop at 16 fps = 2 env steps per second (the viewer's 1x).

usage: .venv-smac/bin/python scripts/present_demo3.py [--tabs 1,2,3,4] [--out-dir artifacts/videos]
"""
import argparse
import json
import math
import textwrap
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SRC = Path("artifacts/battle_replay/demo3_video")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
ALLY, ENEMY, INK, MUTED, BAD, SHIELD, HP = (67, 209, 200), (242, 162, 58), (223, 231, 241), (135, 150, 171), (255, 107, 94), (127, 178, 255), (111, 220, 140)
BG, PANEL, LINE = (11, 16, 24), (18, 26, 38), (36, 50, 74)
R = {"stalker": 0.62, "zealot": 0.5, "colossus": 1.0}
SHOW_BUBBLES = False  # the right panel lists the messages; bubbles hide the units in close fights
HEIGHT = {"zealot": 1.0, "stalker": 1.3, "colossus": 3.6}


def font(path, size):
    return ImageFont.truetype(path, size)


F = {"h1": font(BOLD, 34), "h2": font(BOLD, 22), "b": font(FONT, 22), "s": font(FONT, 19), "lab": font(BOLD, 19),
     "m": font(MONO, 20), "bub": font(FONT, 17), "big": font(BOLD, 64)}


def load_data():
    h = (SRC / "index.html").read_text()
    i = h.index("const DATA = ") + len("const DATA = ")
    return json.JSONDecoder().raw_decode(h[i:])[0]


def proj(cam, X, Y):
    f, D, phi, ox, oy, ty = cam
    dX, dY = X - 16, Y - ty
    z = D + dY * math.cos(phi)
    return ox + f * dX / z, oy - f * dY * math.sin(phi) / z, f / z


def deaths(frames):
    out = []
    for i in range(len(frames) - 1):
        for side, key in (("al", "allies"), ("en", "enemies")):
            nxt = {u["id"]: u for u in frames[i + 1]["u"][key]}
            for u in frames[i]["u"][key]:
                if u["hp"] > 0 and nxt[u["id"]]["hp"] <= 0:
                    out.append((i + 1, side, u["id"], u["type"]))
    return out


def overlay(img, g, L, f, types):
    d = ImageDraw.Draw(img, "RGBA")
    bubbles = []
    for side in ("en", "al"):
        for k, tr in g["track"][side].items():
            v = tr[min(L, len(tr) - 1)]
            if v[2] <= 0:
                continue
            typ = types[side][k]
            x, y, s = proj(g["cam"], v[0], v[1])
            col = ALLY if side == "al" else ENEMY
            hid = side == "en" and f.get("vis") is not None and int(k) not in f["vis"]
            a = 115 if hid else 255
            r = R[typ] * s * 1.15
            d.ellipse([x - r, y - r * 0.55, x + r, y + r * 0.55], outline=col + (a,), width=2)
            ly = y - HEIGHT[typ] * s * 0.95 - 12
            txt = ("a" if side == "al" else "e") + k + ("?" if hid else "")
            w = d.textlength(txt, font=F["lab"]) + 12
            d.rounded_rectangle([x - w / 2, ly - 25, x + w / 2, ly], 5, fill=(7, 11, 17, int(a * 0.8)), outline=col + (a,), width=2)
            d.text((x, ly - 12), txt, font=F["lab"], fill=col + (a,), anchor="mm")
            u = next(u for u in f["u"]["allies" if side == "al" else "enemies"] if str(u["id"]) == k)
            bw = max(w, 34)
            d.rectangle([x - bw / 2 - 1, ly + 2, x + bw / 2 + 1, ly + 11], fill=(5, 8, 12, int(a * 0.85)))
            if u["sh_max"]:
                d.rectangle([x - bw / 2, ly + 3, x - bw / 2 + bw * v[3] / u["sh_max"], ly + 5], fill=SHIELD + (a,))
            fr = v[2] / u["hp_max"]
            d.rectangle([x - bw / 2, ly + 7, x - bw / 2 + bw * fr, ly + 9], fill=(HP if fr > .5 else (233, 211, 106) if fr > .25 else BAD) + (a,))
            o = f["a"].get(k) if side == "al" else None
            if o and o.get("say"):
                bubbles.append((x, ly - 30, o["say"]))
    for x, y, txt in (bubbles if SHOW_BUBBLES else []):
        t = txt if len(txt) <= 34 else txt[:33] + "…"
        w = d.textlength(t, font=F["bub"]) + 16
        bx = min(max(x, w / 2 + 4), 1076 - w / 2)
        by = max(y - 27, 4)
        d.rounded_rectangle([bx - w / 2, by, bx + w / 2, by + 27], 7, fill=(10, 15, 23, 220), outline=ALLY, width=2)
        d.text((bx, by + 13), t, font=F["bub"], fill=INK, anchor="mm")


def panel(img, g, i, dl):
    d = ImageDraw.Draw(img)
    x0, W = 1104, 1920 - 1104 - 24
    f, N, s = g["frames"][i], len(g["frames"]), g["summary"]
    d.text((x0, 28), g["label"], font=F["h1"], fill=INK)
    sub = "Our squad (teal labels) vs built-in AI (orange)" + ("" if g["chat"] else " · no communication")
    d.text((x0, 76), sub, font=F["s"], fill=MUTED)
    alive_a = sum(u["hp"] > 0 for u in f["u"]["allies"])
    alive_e = sum(u["hp"] > 0 for u in f["u"]["enemies"])
    d.text((x0, 112), f"Step {f['t']:>2} / {N - 1}", font=F["h2"], fill=INK)
    d.text((x0 + 230, 112), f"ours {alive_a}/5", font=F["h2"], fill=ALLY)
    d.text((x0 + 390, 112), f"enemy {alive_e}/5", font=F["h2"], fill=ENEMY)
    d.rectangle([x0, 150, x0 + W * i / max(N - 1, 1), 154], fill=ALLY)
    d.rectangle([x0 + W * i / max(N - 1, 1), 150, x0 + W, 154], fill=LINE)
    y = 176
    d.text((x0, y), "ORDERS THIS STEP", font=F["s"], fill=MUTED); y += 32
    for u in f["u"]["allies"]:
        o = f["a"].get(str(u["id"]))
        if u["hp"] <= 0:
            act, col = "destroyed", MUTED
        elif o:
            act = o["act"].replace("attack_", "attack ")
            col = BAD if not o["valid"] else (ENEMY if act.startswith("attack") else INK)
        else:
            act, col = "—", MUTED
        d.text((x0, y), f"a{u['id']}", font=F["h2"], fill=ALLY if u["hp"] > 0 else MUTED)
        d.text((x0 + 52, y + 2), u["type"], font=F["b"], fill=MUTED)
        d.text((x0 + 190, y + 2), act + ("  (illegal)" if o and not o["valid"] and u["hp"] > 0 else ""), font=F["m"], fill=col)
        y += 34
    y += 14
    d.text((x0, y), "TEAM CHAT THIS STEP" if g["chat"] else "TEAM CHAT", font=F["s"], fill=MUTED); y += 32
    if not g["chat"]:
        d.text((x0, y), "C0: the units cannot send messages.", font=F["b"], fill=MUTED); y += 34
    else:
        said = [(k, o["say"]) for k, o in sorted(f["a"].items()) if o.get("say")]
        if not said:
            d.text((x0, y), "Nobody spoke on this step." if f["a"] else "—", font=F["b"], fill=MUTED); y += 34
        for k, msg in said:
            d.text((x0, y), f"a{k}", font=F["h2"], fill=ALLY)
            for j, line in enumerate(textwrap.wrap(msg, 52)[:2]):
                d.text((x0 + 52, y + 2), line, font=F["b"], fill=INK); y += 30
            y += 4
        d.text((x0, y), "Teammates read these in their next prompt.", font=F["s"], fill=MUTED); y += 30
    y = max(y + 14, 760)
    d.text((x0, y), "KILL FEED", font=F["s"], fill=MUTED); y += 32
    ev = [e for e in dl if e[0] <= i][::-1][:7]
    if not ev:
        d.text((x0, y), "No losses yet.", font=F["b"], fill=MUTED)
    for st, side, uid, typ in ev:
        d.text((x0, y), f"s{st:>2}", font=F["m"], fill=MUTED)
        d.text((x0 + 70, y), f"{'a' if side == 'al' else 'e'}{uid} {typ} destroyed", font=F["b"], fill=ALLY if side == "al" else ENEMY)
        y += 30
    if i == N - 1:
        w = s["winner"]
        txt = "OUR SQUAD WINS" if w == "A" else "BUILT-IN AI WINS" if w == "B" else "DRAW"
        d.text((540, 1000), txt, font=F["big"], fill=ALLY if w == "A" else ENEMY, anchor="mm", stroke_width=3, stroke_fill=(0, 0, 0))


def render(g, k, out):
    frames, off, fps = g["frames"], g["offset"], g["fps"]
    types = {side: {str(u["id"]): u["type"] for u in frames[0]["u"][key]} for side, key in (("al", "allies"), ("en", "enemies"))}
    dl = deaths(frames)
    last = off + 8 * (len(frames) - 1)
    w = imageio.get_writer(out, fps=fps, codec="libx264", quality=6, macro_block_size=8, pixelformat="yuv420p",
                           output_params=["-movflags", "+faststart"])
    n = 0
    for L, img in enumerate(imageio.get_reader(SRC / g["video"])):
        if L < off or L > last:
            continue
        i = (L - off) // 8
        canvas = Image.new("RGB", (1920, 1080), BG)
        v = Image.fromarray(img).convert("RGBA")
        overlay(v, g, L, frames[i], types)
        canvas.paste(v.convert("RGB"), (0, 0))
        ImageDraw.Draw(canvas).rectangle([1080, 0, 1919, 1079], fill=PANEL)
        panel(canvas, g, i, dl)
        arr = np.asarray(canvas)
        reps = fps if L == off else 3 * fps if L == last else 1  # hold the first frame 1 s, the last 3 s
        for _ in range(reps):
            w.append_data(arr)
            n += 1
    w.close()
    print(f"wrote {out}: {n} frames ({n / fps:.1f} s), {Path(out).stat().st_size / 1e6:.1f} MB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tabs", default="1,2,3,4")
    p.add_argument("--out-dir", default="artifacts/videos")
    a = p.parse_args()
    data = load_data()
    for t in map(int, a.tabs.split(",")):
        render(data["games"][t - 1], t, f"{a.out_dir}/present_{t}.mp4")


if __name__ == "__main__":
    main()
