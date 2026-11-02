"""Builds index.html from template.html + app.js + the infographic guide below.

Run:  python3 src/build.py
"""
import math, pathlib, html

ROOT = pathlib.Path(__file__).resolve().parent
OUT = ROOT.parent / "index.html"
_mid = [0]


def esc(s):
    return html.escape(str(s), quote=True)


def svg(w, h, body, label):
    _mid[0] += 1
    m = f"ah{_mid[0]}"
    body = body.replace("MARK", m)
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><marker id="{m}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
            f'<path d="M0,0 L10,5 L0,10 z" class="f-mu"/></marker></defs>{body}</svg>')


def rect(x, y, w, h, cls="f-bg", rx=6, extra=""):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" class="{cls}" {extra}/>'


def text(x, y, s, cls="", anchor="start", extra=""):
    return f'<text x="{x:.1f}" y="{y:.1f}" class="{cls}" text-anchor="{anchor}" {extra}>{esc(s)}</text>'


def arrow(x1, y1, x2, y2, dash=False):
    d = ' stroke-dasharray="4 4"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" class="s-mu" stroke-width="1.6"{d} marker-end="url(#MARK)"/>'


def pathA(d, dash=False):
    dd = ' stroke-dasharray="4 4"' if dash else ""
    return f'<path d="{d}" class="s-mu f-none" stroke-width="1.6"{dd} marker-end="url(#MARK)"/>'


def box(x, y, w, h, title, sub="", accent=False):
    out = rect(x, y, w, h, "f-ac" if accent else "f-bg", 8)
    tc = "h on-ac" if accent else "h"
    sc = "sm on-ac" if accent else "sm mu"
    if sub:
        out += text(x + w / 2, y + h / 2 - 3, title, tc, "middle")
        out += text(x + w / 2, y + h / 2 + 14, sub, sc, "middle")
    else:
        out += text(x + w / 2, y + h / 2 + 5, title, tc, "middle")
    return out


def poly(pts, cls, extra=""):
    p = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    return f'<polygon points="{p}" class="{cls}" {extra}/>'


# ---------- 1. Big picture ----------
def fig_overview():
    b = ""
    b += box(16, 70, 112, 60, "Board", "the position")
    b += box(162, 70, 150, 60, "Network", "19 residual blocks")
    b += box(346, 26, 136, 52, "Policy p", "odds for each move")
    b += box(346, 122, 136, 52, "Value v", "−1 loss … +1 win")
    b += box(516, 70, 148, 60, "Tree search", "800 simulations", True)
    b += box(516, 186, 148, 44, "Play the move")
    b += box(162, 186, 320, 44, "Self-play games become training data")
    b += arrow(128, 100, 160, 100) + arrow(312, 90, 344, 56) + arrow(312, 110, 344, 146)
    b += arrow(482, 52, 514, 88) + arrow(482, 148, 514, 112)
    b += arrow(590, 130, 590, 184) + arrow(516, 208, 484, 208) + arrow(237, 186, 237, 132)
    b += pathA("M 590 70 C 590 8, 237 8, 237 68", dash=True)
    b += text(414, 16, "asks the network about every new position", "sm mu", "middle")
    b += text(246, 162, "weights updated", "sm mu")
    return svg(680, 240, b, "AlphaZero's loop: the board goes into the network, which outputs a policy and a value; the tree search uses them to pick a move; self-play games train the network.")


# ---------- 2. Input encoding ----------
def fig_input():
    b = text(16, 30, "Square e4", "h")
    names = ["White pawn", "White knight", "White bishop", "White rook", "White queen", "White king",
             "Black pawn", "Black knight", "Black bishop", "Black rook", "Black queen", "Black king"]
    for i, n in enumerate(names):
        y = 46 + i * 15.5
        on = i == 0
        b += rect(16, y, 12, 11, "f-ac" if on else "f-bg s-gr", 2, 'stroke-width="1"')
        b += text(36, y + 9.5, n, "sm" if on else "sm mu")
        b += text(150, y + 9.5, "1" if on else "0", "sm" if on else "sm mu", "end")
    b += text(16, 252, "one-hot: one 1 per square", "sm mu")
    # stack
    x0 = 190
    for i in range(119):
        if i < 112:
            t, j = divmod(i, 14)
            cls = "f-w" if j < 6 else ("f-b" if j < 12 else "f-mu")
            op = 1 - t * 0.09
            x = x0 + i * 3.1 + t * 6
        else:
            cls, op = "f-c", 1
            x = x0 + 111 * 3.1 + 7 * 6 + 10 + (i - 112) * 4
        b += poly([(x, 222), (x + 26, 196), (x + 26, 46), (x, 72)], f"{cls} s-sf", f'opacity="{op:.2f}" stroke-width="0.6"')
    gx = lambda t: x0 + t * (14 * 3.1 + 6)
    b += text(gx(0) + 30, 244, "now", "sm", "middle")
    b += text(gx(1) + 30, 244, "T−1", "sm mu", "middle")
    b += text(gx(7) + 30, 244, "T−7", "sm mu", "middle")
    sx = x0 + 111 * 3.1 + 7 * 6 + 10
    b += text(sx + 26, 244, "state", "sm mu", "middle")
    b += text(gx(0), 30, "14 × 8 moments + 7 state sheets = 119", "h")
    return svg(680, 262, b, "One square encoded as twelve yes/no values, and the full stack of 119 sheets: 14 per moment for 8 moments, plus 7 game-state sheets.")


# ---------- 3. Convolution ----------
def sheet(bx, by, k, cell=13, n=8, sh=6, dk=(9, -7)):
    """Return mapping function for a sheered sheet k."""
    ox, oy = bx + k * dk[0], by + k * dk[1]
    def P(u, v):
        return (ox + u * cell + v * sh, oy - v * cell * 0.62)
    return P


def grid_sheet(P, cls_fill, n=8, hi=None, hi_cls="f-ac", op=1.0):
    out = poly([P(0, 0), P(n, 0), P(n, n), P(0, n)], f"{cls_fill} s-gr", f'stroke-width="1" opacity="{op}"')
    for i in range(1, n):
        a, b2 = P(i, 0), P(i, n)
        out += f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b2[0]:.1f}" y2="{b2[1]:.1f}" class="s-gr" stroke-width="0.6" opacity="{op}"/>'
        a, b2 = P(0, i), P(n, i)
        out += f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b2[0]:.1f}" y2="{b2[1]:.1f}" class="s-gr" stroke-width="0.6" opacity="{op}"/>'
    if hi:
        for (u0, v0, u1, v1) in hi:
            out += poly([P(u0, v0), P(u1, v0), P(u1, v1), P(u0, v1)], hi_cls, f'opacity="{0.85*op:.2f}"')
    return out


def fig_conv():
    b = ""
    K = 6
    for k in range(K - 1, -1, -1):
        P = sheet(24, 210, k)
        b += grid_sheet(P, "f-sf", hi=[(3, 3, 6, 6)], op=1.0 if k == 0 else 0.92)
    b += text(24, 236, "input · 8×8×119 (6 of 119 drawn)", "sm mu")
    b += text(24, 24, "one filter = 3×3 × 119 weights", "h")
    b += arrow(250, 140, 330, 140)
    b += text(290, 118, "multiply, add,", "sm mu", "middle")
    b += text(290, 132, "+ bias, ReLU", "sm mu", "middle")
    b += text(290, 160, "1,071 weights", "sm", "middle")
    P = sheet(340, 210, 0)
    b += grid_sheet(P, "f-sf", hi=[(4, 4, 5, 5)])
    b += text(340, 236, "1 filter → 1 sheet", "sm")
    b += arrow(500, 140, 540, 140)
    for k in range(7, -1, -1):
        P = sheet(548, 210, k, cell=7, sh=3.5, dk=(6, -5))
        b += grid_sheet(P, "f-sf" if k else "f-ac", op=0.9)
    b += text(548, 236, "256 filters → 8×8×256", "sm")
    return svg(680, 250, b, "A 3 by 3 window reaches through all 119 input sheets and produces one number per square; one filter makes one output sheet, 256 filters make 256 sheets.")


# ---------- 4. Residual block ----------
def fig_residual():
    b = ""
    y = 64
    b += f'<circle cx="34" cy="{y+22}" r="18" class="f-bg"/>' + text(34, y + 27, "x", "h", "middle")
    b += box(78, y, 150, 44, "3×3 conv", "BN · ReLU")
    b += box(258, y, 150, 44, "3×3 conv", "BN")
    b += f'<circle cx="452" cy="{y+22}" r="18" class="f-ac"/>' + text(452, y + 28, "+", "h on-ac", "middle")
    b += box(496, y, 86, 44, "ReLU")
    b += text(640, y + 27, "out", "h", "middle")
    b += arrow(52, y + 22, 76, y + 22) + arrow(228, y + 22, 256, y + 22) + arrow(408, y + 22, 432, y + 22)
    b += arrow(470, y + 22, 494, y + 22) + arrow(582, y + 22, 620, y + 22)
    b += pathA(f"M 34 {y+4} C 34 20, 452 20, 452 {y+2}")
    b += text(243, 14, "skip connection: the block's input is added back", "sm mu", "middle")
    # receptive field bars
    b += text(16, 152, "How far one square sees", "h")
    rows = [("first conv", 3), ("after block 1", 7), ("after block 2", 11), ("after block 3", 15)]
    for i, (n, s) in enumerate(rows):
        yy = 166 + i * 20
        w = s / 15 * 430
        b += text(16, yy + 12, n, "sm mu")
        b += rect(130, yy, w, 14, "f-ac" if s >= 15 else "f-gr", 3)
        b += text(130 + w + 8, yy + 12, f"{s}×{s}" + ("  whole board from any square" if s >= 15 else ""), "sm")
    return svg(680, 254, b, "A residual block: two 3 by 3 convolutions with the input added back, and a chart of how the visible patch grows from 3 by 3 to 15 by 15.")


# ---------- 5. Bishop relay ----------
def fig_relay():
    b = ""
    cell = 26
    ox, oy = 20, 16
    for r in range(8):
        for c in range(8):
            cls = "f-gr" if (r + c) % 2 == 0 else "f-bg"
            b += rect(ox + c * cell, oy + (7 - r) * cell, cell, cell, cls, 0)
    def ctr(r, c):
        return ox + c * cell + cell / 2, oy + (7 - r) * cell + cell / 2
    bx, by = ctr(0, 2)
    b += text(bx, by + 7, "♝", "h", "middle", 'style="font-size:20px"')
    for i, (r, c) in enumerate([(1, 3), (2, 4), (3, 5)]):
        x, y = ctr(r, c)
        b += f'<circle cx="{x:.1f}" cy="{y:.1f}" r="10" class="f-ac"/>' + text(x, y + 4, str(i + 1), "sm on-ac", "middle")
    x, y = ctr(4, 6)
    b += text(x, y + 7, "♞", "h", "middle", 'style="font-size:20px"')
    x2, y2 = ctr(5, 7)
    b += f'<line x1="{x2-8:.1f}" y1="{y2-8:.1f}" x2="{x2+8:.1f}" y2="{y2+8:.1f}" class="s-mu" stroke-width="2"/><line x1="{x2+8:.1f}" y1="{y2-8:.1f}" x2="{x2-8:.1f}" y2="{y2+8:.1f}" class="s-mu" stroke-width="2"/>'
    tx = 250
    b += text(tx, 40, "A ray sheet for “bishop shining north-east”", "h")
    lines = ["Each layer looks only at the south-west neighbour.",
             "The ray reaches a square if:",
             "  · a bishop or queen stands on that neighbour, or",
             "  · the ray reached the neighbour and it is empty.",
             "",
             "Numbers on the board = the layer where the ray arrives.",
             "The knight on g5 stops it: h6 is never reached.",
             "",
             "7 layers cross the board. AlphaZero has about 39."]
    for i, l in enumerate(lines):
        b += text(tx, 66 + i * 18, l, "sm" if i in (1, 5, 8) else "sm mu", "start", 'xml:space="preserve"')
    return svg(680, 240, b, "A bishop's attack travels one square per layer along the diagonal and stops at the first blocking piece.")


# ---------- 6. Policy ----------
def fig_policy():
    b = text(16, 26, "73 sheets per starting square", "h")
    W = 640
    segs = [(56, "56 slides · 8 directions × 1–7 squares", "f-ac"), (8, "8 knight", "f-w"), (9, "9 promo", "f-b")]
    x = 16
    for n, lab, cls in segs:
        w = n / 73 * W
        b += rect(x, 38, w - 2, 30, cls, 4)
        b += text(x + 6, 58, lab, "sm on-ac" if cls == "f-ac" else "sm on-ac")
        x += w
    b += text(16, 96, "× 64 starting squares = 4,672 slots", "sm")
    # funnel
    b += box(16, 118, 170, 52, "4,672 raw scores", "one per slot")
    b += box(226, 118, 170, 52, "mask illegal moves", "≈ 20–40 remain")
    b += box(436, 118, 120, 52, "softmax", "scores → %", True)
    b += arrow(186, 144, 224, 144) + arrow(396, 144, 434, 144)
    # bars
    ex = [("e4", 2.0, .67), ("d4", 1.2, .30), ("a3", -1.0, .03)]
    b += text(16, 200, "example", "sm mu")
    for i, (m, lg, p) in enumerate(ex):
        yy = 210 + i * 22
        b += text(16, yy + 12, m, "sm")
        b += text(52, yy + 12, f"logit {lg:+.1f}", "sm mu")
        b += rect(150, yy, max(3, p * 420), 15, "f-ac", 3)
        b += text(158 + p * 420, yy + 12, f"{p:.0%}", "sm")
    return svg(680, 280, b, "The policy head scores 4,672 move slots, masks the illegal ones, and turns the rest into probabilities with softmax.")


# ---------- 7. Value ----------
def fig_value():
    b = ""
    items = [("8×8×256", "tower output"), ("1×1 conv", "1 filter"), ("64", "numbers"), ("256 units", "fully connected"), ("1", "number"), ("tanh", "squash")]
    x = 12
    for i, (t, s) in enumerate(items):
        w = 98
        b += box(x, 20, w, 50, t, s, accent=(i == 5))
        if i < 5:
            b += arrow(x + w, 45, x + w + 12, 45)
        x += w + 12
    b += text(16, 104, "1×1 conv: mixes the 256 features on each square, no neighbours", "sm mu")
    b += text(16, 122, "fully connected: every unit sees all 64 squares at once", "sm mu")
    # tanh plot
    ox, oy, W, H = 60, 150, 260, 96
    b += f'<line x1="{ox}" y1="{oy+H/2}" x2="{ox+W}" y2="{oy+H/2}" class="s-gr"/><line x1="{ox+W/2}" y1="{oy}" x2="{ox+W/2}" y2="{oy+H}" class="s-gr"/>'
    pts = []
    for i in range(61):
        z = -3 + i * 0.1
        pts.append(f"{ox + (z+3)/6*W:.1f},{oy + H/2 - math.tanh(z)*H/2:.1f}")
    b += f'<polyline points="{" ".join(pts)}" class="s-ac f-none" stroke-width="2.5"/>'
    b += text(ox - 8, oy + 8, "+1", "sm mu", "end") + text(ox - 8, oy + H, "−1", "sm mu", "end")
    b += text(ox + W, oy + H / 2 + 16, "raw score", "sm mu", "end")
    tx = 360
    rows = [("+1", "side to move wins"), ("0", "draw"), ("−1", "side to move loses"), ("+0.2", "expected score ≈ 60%")]
    for i, (v, d) in enumerate(rows):
        b += text(tx, 168 + i * 22, v, "sm", "end" if False else "start")
        b += text(tx + 50, 168 + i * 22, d, "sm mu")
    return svg(680, 262, b, "The value head: a 1 by 1 convolution, a 256-unit fully connected layer, and tanh, giving one number from −1 to +1.")


# ---------- 8. Search ----------
def fig_search():
    b = ""
    b += box(30, 24, 200, 56, "1  Select", "follow highest Q + U")
    b += box(450, 24, 200, 56, "2  Expand & evaluate", "ask the network: p and v", True)
    b += box(450, 160, 200, 56, "3  Back up", "add v to every node above")
    b += box(30, 160, 200, 56, "4  Repeat", "800 times per move")
    b += arrow(230, 52, 448, 52) + arrow(550, 80, 550, 158) + arrow(450, 188, 232, 188) + arrow(130, 160, 130, 82)
    # mini tree
    nodes = {(340, 96): "root", (300, 132): "", (380, 132): "", (360, 168): "", (400, 168): ""}
    edges = [((340, 96), (300, 132)), ((340, 96), (380, 132)), ((380, 132), (360, 168)), ((380, 132), (400, 168))]
    for (a, c) in edges:
        b += f'<line x1="{a[0]}" y1="{a[1]}" x2="{c[0]}" y2="{c[1]}" class="s-gr" stroke-width="1.5"/>'
    for (x, y) in nodes:
        b += f'<circle cx="{x}" cy="{y}" r="8" class="{"f-ac" if (x, y) in ((340, 96), (380, 132), (400, 168)) else "f-gr"}"/>'
    b += text(340, 250, "Then play the move visited most.  U ∝ P · √N ÷ (1 + n)", "sm", "middle")
    return svg(680, 262, b, "Monte Carlo tree search: select by Q plus U, expand and evaluate with the network, back up the value, repeat 800 times, then play the most visited move.")


# ---------- 9. Training ----------
def fig_train():
    b = ""
    b += box(12, 30, 150, 56, "Self-play game", "search picks each move")
    b += box(190, 30, 190, 56, "Store every position", "board · visit % π · result z")
    b += box(408, 30, 150, 56, "Compute loss", "", True)
    b += box(586, 30, 82, 56, "New", "weights")
    b += arrow(162, 58, 188, 58) + arrow(380, 58, 406, 58) + arrow(558, 58, 584, 58)
    b += pathA("M 627 86 C 627 140, 87 140, 87 88")
    b += text(340, 152, "the improved network plays the next games", "sm mu", "middle")
    b += text(340, 186, "loss = (z − v)²  −  π · log p  +  c‖θ‖²", "h", "middle")
    b += text(170, 206, "value error", "sm mu", "middle") + text(340, 206, "policy error", "sm mu", "middle") + text(500, 206, "keep weights small", "sm mu", "middle")
    return svg(680, 218, b, "The training loop: self-play games produce positions with search visit counts and results, which drive the loss that updates the weights.")


# ---------- 10. Real filters ----------
F2 = {"rook sheet": [[0.00, 0.02, 0.00], [0.35, 1.55, 0.00], [0.02, 0.03, 0.02]],
      "queen sheet": [[0.01, 0.03, 0.01], [0.35, 0.98, -0.02], [0.03, 0.03, 0.03]]}
F29 = {"bishop sheet": [[0.31, 0.01, 0.28], [0.03, 1.05, 0.01], [0.13, -0.01, 0.32]],
       "queen sheet": [[0.22, 0.15, 0.32], [0.01, 0.63, 0.04], [0.07, 0.04, 0.19]]}


def fig_real():
    b = ""
    def kgrid(x, y, w, title):
        out = text(x, y - 8, title, "sm mu")
        for r in range(3):
            for c in range(3):
                v = w[r][c]
                a = min(1, abs(v) / 1.1)
                out += rect(x + c * 38, y + r * 38, 36, 36, "f-bg", 3)
                if a > 0.02:
                    out += rect(x + c * 38, y + r * 38, 36, 36, "f-ac" if v > 0 else "f-w", 3, f'opacity="{max(0.12,a):.2f}"')
                out += text(x + c * 38 + 18, y + r * 38 + 22, f"{v:+.2f}", "sm on-ac" if (v > 0 and a > 0.55) else "sm", "middle")
        return out
    b += text(16, 24, "Filter 2 · “a rook-mover is here”", "h")
    b += kgrid(16, 54, F2["rook sheet"], "their rook")
    b += kgrid(150, 54, F2["queen sheet"], "their queen")
    b += text(356, 24, "Filter 29 · “a diagonal-mover is near”", "h")
    b += kgrid(356, 54, F29["bishop sheet"], "their bishop")
    b += kgrid(490, 54, F29["queen sheet"], "their queen")
    b += text(16, 196, "top row = toward the opponent · rook and queen share a pattern; bishop and queen share another", "sm mu")
    return svg(680, 210, b, "Two real learned first-layer filters from the open Maia network: one responds to rooks and queens, one to bishops and queens with weight on the diagonal neighbours.")


# ---------- 11. Result ----------
def fig_result():
    b = text(16, 24, "1,000-game match vs Stockfish 8 (Science, 2018)", "h")
    tot = 1000
    W = 640
    x = 16
    for n, lab, cls in [(155, "155 wins", "f-ac"), (839, "839 draws", "f-gr"), (6, "6 losses", "f-b")]:
        w = n / tot * W
        b += rect(x, 40, max(w, 3), 34, cls, 0)
        x += w
    b += text(20, 62, "155 wins", "h on-ac")
    b += text(16 + 155 / tot * W + 10, 62, "839 draws", "h")
    b += text(16 + W, 96, "6 losses", "sm", "end")
    return svg(680, 108, b, "AlphaZero's 1,000-game match against Stockfish 8: 155 wins, 839 draws, 6 losses.")


SECTIONS = [
    ("The whole idea", "One network, one search, no human games",
     "AlphaZero learns chess from the rules alone. A single neural network looks at a position and gives two answers: which moves look promising, and who is winning. A tree search uses those answers to choose a move. Every game it plays against itself becomes training data that makes the network better.",
     "<b>Principle.</b> The search makes the network's guesses better; training makes the network match the search. Each improves the other.",
     fig_overview(), False),
    ("Input planes", "The board becomes 119 yes-or-no sheets",
     "Each piece type of each colour gets its own 8×8 sheet: 1 where that piece stands, 0 elsewhere. Twelve piece sheets plus two repetition sheets describe one moment; AlphaZero stacks the current moment and the seven before it, then adds seven sheets for side to move, castling rights and move counters.",
     "<b>Why sheets, not numbers?</b> Labelling a knight “2” and a bishop “3” would invent an order and arithmetic that chess doesn't have. Separate sheets let every filter treat each piece independently.",
     fig_input(), False),
    ("Convolution", "A 3×3 window that reaches through all the sheets",
     "A filter has a 3×3 grid of weights for every input sheet, 1,071 weights in all. At each square it multiplies those weights by the values underneath, adds everything into one number, adds a bias and keeps it only if it's positive. Sliding over all 64 squares gives one new 8×8 sheet. The first layer has 256 filters, so it outputs 8×8×256.",
     "<b>Principle.</b> The same filter is used on every square, because a pattern means the same thing on the c-file as on the g-file. Nobody designs the weights: training finds them.",
     fig_conv(), False),
    ("Residual tower", "19 blocks that build bigger ideas",
     "Each block runs two 3×3 convolutions with batch normalisation, then adds its own input back before the final ReLU. Stacking blocks lets each square's features depend on a wider and wider patch of the board. After three blocks, every square can be influenced by every other square.",
     "<b>Why the skip connection?</b> Adding the input back means a block only has to learn a correction. That keeps information flowing through a very deep network and makes it trainable.",
     fig_residual(), False),
    ("Long-range pieces", "How 3×3 filters can see a bishop across the board",
     "No single 3×3 filter can see a bishop five squares away. But a stack of layers can pass a ray along one square per layer, stopping at the first piece in the way, just like the real piece. Real trained networks contain this kind of first step: filters that respond to a bishop or queen on the diagonal neighbours.",
     "<b>Principle.</b> Depth turns local rules into long-range knowledge. Small filters, many layers.",
     fig_relay(), False),
    ("Policy head", "A score for every possible move",
     "The policy head turns the tower's output into 73 sheets. A move is a starting square plus one sheet that says which way and how far it goes, so there are 4,672 slots. Illegal slots are thrown away and softmax turns the rest into probabilities that add up to 100%.",
     "<b>How it's used.</b> The policy is not the final choice. It tells the search which moves deserve a closer look. It's trained to match how often the search actually visited each move.",
     fig_policy(), False),
    ("Value head", "One number: who is winning?",
     "The value head shrinks the tower's output to a single sheet with a 1×1 convolution, feeds those 64 numbers into a 256-unit fully connected layer, and squashes the result with tanh into the range −1 to +1, from the point of view of the side to move.",
     "<b>How it's trained.</b> Every position of a finished self-play game is labelled with the final result, and the value head learns to predict it. This replaced the random playouts older programs used.",
     fig_value(), False),
    ("Tree search", "Thinking ahead with the network's help",
     "Monte Carlo tree search grows a tree of possible continuations. It follows the most promising branch, asks the network about the new position, and passes the value back up. Q is how good a move has proved so far; U favours moves the policy likes that haven't been tried much.",
     "<b>Selective, not exhaustive.</b> AlphaZero examined tens of thousands of positions per second, about a thousand times fewer than Stockfish, and still played stronger chess.",
     fig_search(), False),
    ("Training", "Learning entirely from self-play",
     "Training plays games of the newest network against itself, stores each position with the search's visit percentages and the final result, and nudges every weight to reduce the combined loss.",
     None,
     fig_train(), True),
    ("Inside a trained network", "What real filters look like",
     "AlphaZero's weights were never released, but Maia, an open network with the same design, lets us look inside. Some first-layer filters are crisp: one fires for rooks and queens together, another for bishops and queens with weight on the diagonal neighbours. Most are blurry blends with no human name.",
     "<b>Nobody told it</b> that queens move like rooks and bishops. Training discovered it. Research on AlphaZero itself (McGrath et al., <i>PNAS</i>, 2022) found concepts like material, mobility and king safety encoded in its layers.",
     fig_real(), False),
    ("The result", "Superhuman chess in a day of training",
     "In the final match reported in <i>Science</i>, AlphaZero played 1,000 games against Stockfish 8, then one of the strongest engines in the world.",
     None,
     fig_result(), True),
]


def guide_html():
    out = ['<div class="wrap">',
           '<div class="ghead"><div><p class="eyebrow">An illustrated guide</p><h1>How AlphaZero thinks</h1>'
           '<p>Eleven steps from a chess position to a move, and from a blank network to superhuman play.</p></div>'
           '<a class="viewlink" href="#explore">Open the 3D explorer</a></div>']
    for i, (eyebrow, h, p, rule, fig, wide) in enumerate(SECTIONS):
        num = f"{i+1:02d} · {eyebrow}"
        txt = f'<div class="txt"><span class="num">{esc(num)}</span><h2>{esc(h)}</h2><p>{p}</p>'
        if rule:
            txt += f'<p class="rule">{rule}</p>'
        if eyebrow == "Training":
            txt += ('<div class="stats"><div><b>44M</b><span>self-play games</span></div><div><b>700k</b><span>training steps</span></div>'
                    '<div><b>4,096</b><span>positions per batch</span></div><div><b>~9 h</b><span>to pass Stockfish</span></div></div>')
        txt += "</div>"
        cls = "gs wide" if wide else "gs"
        out.append(f'<section class="{cls}">{txt}<div class="fig">{fig}</div></section>')
    out.append('<p class="gfoot">Sources: Silver et al., “A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play,” <i>Science</i> 362 (2018); '
               'McGrath et al., “Acquisition of chess knowledge in AlphaZero,” <i>PNAS</i> 119 (2022); real filter weights from the open '
               '<a href="https://github.com/CSSLab/maia-chess" rel="noopener">Maia Chess</a> network (maia-1900), which uses the Leela Chess Zero architecture. '
               'The 3D explorer runs AlphaZero-shaped layers with random weights.</p>')
    out.append("</div>")
    return "\n".join(out)


def main():
    tpl = (ROOT / "template.html").read_text()
    js = (ROOT / "app.js").read_text()
    page = tpl.replace("<!--GUIDE-->", guide_html()).replace("<!--APPJS-->", js)
    OUT.write_text(page)
    print("wrote", OUT, len(page), "bytes")


if __name__ == "__main__":
    main()
