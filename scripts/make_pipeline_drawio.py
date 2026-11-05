"""Pipeline figure v2 (with mutual information), same visual style as the original trustmi_har_pipeline.drawio."""
import html, re, sys
src = open(sys.argv[1]).read()
st = lambda cid: re.search(rf'<mxCell id="{cid}"(?: value="[^"]*")? style="([^"]*)"', src).group(1)
BOX, SENS, LINE, ARROW = st("c10"), st("c1"), st("c4"), st("c8")
SMALL = BOX.replace("fontSize=16", "fontSize=15")
cells, n = [], [0]

def nid():
    n[0] += 1
    return f"v{n[0]}"

def lab(title, sub=None, ts=16, ss=14):
    s = f'<b style="font-size:{ts}px">{title}</b>'
    return s + (f'<br><span style="font-size:{ss}px">{sub}</span>' if sub else "")

def box(x, y, w, h, value, style=BOX):
    cells.append(f'<mxCell id="{nid()}" value="{html.escape(value, quote=True)}" style="{style}" vertex="1" parent="1">'
                 f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

def seg(x1, y1, x2, y2, arrow=False):
    cells.append(f'<mxCell id="{nid()}" style="{ARROW if arrow else LINE}" edge="1" parent="1"><mxGeometry relative="1" as="geometry">'
                 f'<mxPoint x="{x1}" y="{y1}" as="sourcePoint"/><mxPoint x="{x2}" y="{y2}" as="targetPoint"/></mxGeometry></mxCell>')

def bus(froms, y_from, y_bus, tos, y_to):
    for x in froms: seg(x, y_from, x, y_bus)
    xs = froms + tos; seg(min(xs), y_bus, max(xs), y_bus)
    for x in tos: seg(x, y_bus, x, y_to, arrow=True)

W3 = [(12, 152), (176, 152), (340, 152)]                       # three equal columns
W4 = [(12, 114), (134, 114), (256, 114), (378, 114)]           # four equal columns
W2 = [(12, 231), (261, 231)]
c = lambda col: [x + w / 2 for x, w in col]
# row 0: sensors
for (x, w), s in zip(W3, ["Inertial", "Skeleton", "Video"]):
    box(x, 10, w, 38, lab(s), SENS)
bus(c(W3), 48, 64, c(W3), 84)
# row 1: fused model, witnesses, MI matcher
box(*W3[0][:1], 84, W3[0][1], 60, lab("Fused model f", "all sensors → p, ŷ"))
box(W3[1][0], 84, W3[1][1], 60, lab("Witnesses g<sub>m</sub>", "one sensor each → q<sub>m</sub>"))
box(W3[2][0], 84, W3[2][1], 60, lab("MI matcher h", "sensor pairs → i<sub>mn</sub> (no labels)"))
bus(c(W3), 144, 162, c(W4), 182)
# row 2: four signals
sig = [("s<sub>1</sub>: doubt", "1 − max p"), ("s<sub>2</sub>: disagree", "mean JS(q<sub>m</sub>,q<sub>n</sub>)"),
       ("s<sub>3</sub>: weak support", "1 − min q<sub>m</sub>[ŷ]"), ("s<sub>4</sub>: low MI", "− min i<sub>mn</sub>")]
for (x, w), (t, s) in zip(W4, sig):
    box(x, 182, w, 56, lab(t, s, 15, 13), SMALL)
bus(c(W4), 238, 256, c(W2), 276)
# row 3: trust score and naming
box(W2[0][0], 276, W2[0][1], 56, lab("Trust score T", "4 signals, fitted on validation"))
box(W2[1][0], 276, W2[1][1], 56, lab("Name sensor m*", "odd one out or lowest MI share"))
bus(c(W2), 332, 350, c(W3), 370)
# row 4: decision
for (x, w), (t, s) in zip(W3, [("Use ŷ", "T below τ"), ("Drop m*", "backup without m*"), ("Abstain", "highest T values")]):
    box(x, 370, w, 56, lab(t, s))
head = re.search(r'(<mxfile.*?<root>)', src, re.S).group(1).replace('pageHeight="432"', 'pageHeight="436"')
print(head + '<mxCell id="0"/><mxCell id="1" parent="0"/>' + "".join(cells) + "</root></mxGraphModel></diagram></mxfile>")
