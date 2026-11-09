"""Export the existing figure as native, editable Visio shapes and rich text.

The draft is a self-contained OPC/VSDX package. finalize_visio.ps1 uses the
installed Visio to glue connectors, save a canonical copy, and render it.
No user Visio files are used as templates or modified.
"""
from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from matplotlib.colors import to_hex
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "draw_figure.py"
NS = "http://schemas.microsoft.com/office/visio/2012/main"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
ET.register_namespace("", NS)
ET.register_namespace("r", REL)


def node(parent, tag, attrs=None, text=None):
    e = ET.SubElement(parent, f"{{{NS}}}{tag}", attrs or {})
    if text is not None:
        e.text = str(text)
    return e


def cell(parent, name, value, formula=None):
    attrs = {"N": name, "V": str(value)}
    if formula is not None:
        attrs["F"] = formula
    return node(parent, "Cell", attrs)


def xml_bytes(root):
    return ET.tostring(root, encoding="utf-8", xml_declaration=True)


def rich_math(value):
    """Convert this figure's small math vocabulary to editable text runs."""
    if not (value.startswith("$") and value.endswith("$")):
        return [(value, 0, False)]
    value = value[1:-1]
    cursor = 0
    runs = []
    symbols = {"mu": "μ", "sigma": "σ", "pi": "π", "beta": "β", "to": "→",
               "Vert": "∥", "cdot": "·", "log": "log", "{": "{", "}": "}",
               ";": " ", ",": " ", " ": " "}

    def append(text, position, italic):
        if not text:
            return
        if runs and runs[-1][1:] == (position, italic):
            runs[-1] = (runs[-1][0] + text, position, italic)
        else:
            runs.append((text, position, italic))

    def atom(position, italic):
        nonlocal cursor
        if cursor >= len(value):
            return
        ch = value[cursor]
        cursor += 1
        if ch == "{":
            sequence(position, italic, closing=True)
        elif ch == "\\":
            match = re.match(r"[A-Za-z]+|.", value[cursor:])
            if not match:
                raise ValueError("Incomplete math command")
            command = match.group()
            cursor += len(command)
            if command == "mathrm":
                atom(position, False)
            elif command == "mathbb":
                start = len(runs)
                atom(position, False)
                for j in range(start, len(runs)):
                    runs[j] = (runs[j][0].replace("E", "𝔼"), runs[j][1], runs[j][2])
            elif command == "bar":
                atom(position, italic)
                append("\u0304", position, italic)
            elif command in symbols:
                append(symbols[command], position, False if command == "log" else italic)
            else:
                raise ValueError(f"Unsupported math command: {command}")
        else:
            append(ch, position, italic)

    def sequence(position=0, italic=True, closing=False):
        nonlocal cursor
        while cursor < len(value):
            ch = value[cursor]
            if ch == "}":
                if not closing:
                    raise ValueError("Unmatched math brace")
                cursor += 1
                return
            if ch in "_^":
                cursor += 1
                atom(2 if ch == "_" else 1, italic)
            else:
                atom(position, italic)

    sequence()
    return runs


scope = {"__file__": str(SOURCE)}
source = SOURCE.read_text(encoding="utf-8")
marker = "# Ensure every artist fits on canvas;"
if marker not in source:
    raise ValueError("The figure source's export boundary changed")
exec(compile(source.split(marker)[0], str(SOURCE), "exec"), scope)
fig, ax = scope["fig"], scope["ax"]
W, H = scope["W"], scope["H"]
SCALE = W / fig.get_figwidth()
fig.canvas.draw()
renderer = fig.canvas.get_renderer()

records = []
order = 0


def record(kind, artist, **extra):
    global order
    order += 1
    item = dict(kind=kind, z=artist.get_zorder(), order=order, **extra)
    records.append(item)
    return item


for p in ax.patches:
    color = to_hex(p.get_edgecolor(), keep_alpha=False)
    stroke = float(p.get_linewidth())
    dashed = p.get_linestyle() != "solid"
    if isinstance(p, FancyArrowPatch):
        record("arrow", p, points=p._path_original.vertices.tolist(), color=color,
               weight=stroke, dashed=dashed)
    elif isinstance(p, Circle):
        x, y = p.center
        record("ellipse", p, bounds=[x-p.radius, y-p.radius, 2*p.radius, 2*p.radius],
               fill=to_hex(p.get_facecolor()), color=color, weight=stroke, dashed=dashed)
    elif isinstance(p, (Rectangle, FancyBboxPatch)):
        if isinstance(p, Rectangle):
            x, y = p.get_xy(); width, height = p.get_width(), p.get_height()
        else:
            x, y, width, height = p.get_bbox().bounds
        record("rect", p, bounds=list(map(float, [x, y, width, height])),
               fill=to_hex(p.get_facecolor()), color=color, weight=stroke,
               dashed=dashed, rounded=isinstance(p, FancyBboxPatch))

for line in ax.lines:
    record("line", line, points=list(zip(map(float, line.get_xdata()), map(float, line.get_ydata()))),
           color=to_hex(line.get_color()), weight=float(line.get_linewidth()), dashed=False)

for text in ax.texts:
    bb = text.get_window_extent(renderer)
    points = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
    width, height = abs(points[1, 0]-points[0, 0]), abs(points[1, 1]-points[0, 1])
    if int(text.get_rotation()) % 180 == 90:
        width, height = height, width
    width = max(width*1.12 + 3, 10)
    height = max(height*1.2, text.get_fontsize()*SCALE/72*1.28)
    x, y = map(float, text.get_position())
    cx = x + (width/2 if text.get_ha() == "left" else -width/2 if text.get_ha() == "right" else 0)
    record("text", text, bounds=[cx-width/2, y-height/2, width, height],
           anchor=[x,y], rotation=float(text.get_rotation()), color=to_hex(text.get_color()),
           size=float(text.get_fontsize()), bold=text.get_fontweight() == "bold",
           runs=rich_math(text.get_text()))

# Group each module with its labels and neural-network primitives.
modules = [r for r in records if r["kind"] == "rect" and r["bounds"][2] < 750]
for m in modules:
    m["children"] = []
for r in records:
    r["parent"] = None
    if r in modules or r["kind"] in {"arrow", "rect"}:
        continue
    for m in modules:
        x,y,w,h = m["bounds"]
        if r["kind"] == "line":
            inside = all(x <= px <= x+w and y <= py <= y+h for px,py in r["points"])
        else:
            if r["kind"] == "text":
                px,py = r["anchor"]
            else:
                bx,by,bw,bh = r["bounds"]; px,py = bx+bw/2,by+bh/2
            inside = x <= px <= x+w and y <= py <= y+h
        if inside:
            r["parent"] = m["order"]
            m["children"].append(r)
            break

page_root = ET.Element(f"{{{NS}}}PageContents", {"{http://www.w3.org/XML/1998/namespace}space":"preserve"})
shapes_root = node(page_root, "Shapes")
next_id = 0
stats = {"groups":0,"text_shapes":0,"native_lines":0,"neurons":0,"embedded_images":0}


def new_shape(parent, name, kind="Shape"):
    global next_id
    next_id += 1
    return node(parent, "Shape", {"ID":str(next_id), "NameU":name, "Type":kind,
                                 "LineStyle":"0","FillStyle":"0","TextStyle":"0"})


def transform(shape, bounds, origin=(0,0), rotation=0):
    x,y,w,h = bounds
    vals = {"PinX":(x+w/2-origin[0])/SCALE,
            "PinY":(origin[1]-y-h/2)/SCALE,
            "Width":w/SCALE,"Height":h/SCALE,
            "LocPinX":w/(2*SCALE),"LocPinY":h/(2*SCALE),
            "Angle":math.radians(rotation)}
    for k,v in vals.items(): cell(shape,k,v)


def style(shape, r, fill=False):
    cell(shape,"LineColor",r["color"])
    cell(shape,"LineWeight",r.get("weight",1)/72)
    cell(shape,"LinePattern",2 if r.get("dashed") else 1)
    cell(shape,"FillPattern",1 if fill else 0)
    if fill: cell(shape,"FillForegnd",r["fill"])
    cell(shape,"ShdwPattern",0)


def draw(parent, r, origin=(0,H), module_body=False):
    kind = r["kind"]
    shape = new_shape(parent, f"{kind}_{r['order']}")
    if not module_body:
        r["id"] = int(shape.get("ID"))
    if kind in {"rect","ellipse","text"}:
        transform(shape,r["bounds"],origin,r.get("rotation",0))
    if kind in {"rect","ellipse"}:
        style(shape,r,fill=True)
        if r.get("rounded"): cell(shape,"Rounding",17/SCALE)
        geo=node(shape,"Section",{"N":"Geometry","IX":"0"})
        cell(geo,"NoFill",0); cell(geo,"NoLine",0)
        if kind == "rect":
            for j,(x,y) in enumerate([(0,0),(1,0),(1,1),(0,1),(0,0)],1):
                row=node(geo,"Row",{"T":"RelMoveTo" if j==1 else "RelLineTo","IX":str(j)})
                cell(row,"X",x);cell(row,"Y",y)
        else:
            stats["neurons"] += 1
            w,h=r["bounds"][2]/SCALE,r["bounds"][3]/SCALE
            row=node(geo,"Row",{"T":"Ellipse","IX":"1"})
            for n,v in {"X":w/2,"Y":h/2,"A":w,"B":h/2,"C":w/2,"D":h}.items(): cell(row,n,v)
    elif kind in {"line","arrow"}:
        stats["native_lines"] += 1
        pts=[((x-origin[0])/SCALE,(origin[1]-y)/SCALE) for x,y in r["points"]]
        sx,sy=pts[0];ex,ey=pts[-1];w,h=ex-sx,ey-sy
        for n,v,f in [("BeginX",sx,None),("BeginY",sy,None),("EndX",ex,None),("EndY",ey,None),
                      ("PinX",(sx+ex)/2,"(BeginX+EndX)*0.5"),("PinY",(sy+ey)/2,"(BeginY+EndY)*0.5"),
                      ("Width",w,"EndX-BeginX"),("Height",h,"EndY-BeginY"),
                      ("LocPinX",w/2,"Width*0.5"),("LocPinY",h/2,"Height*0.5"),("Angle",0,None)]:
            cell(shape,n,v,f)
        style(shape,r)
        cell(shape,"EndArrow",4 if kind=="arrow" else 0)
        cell(shape,"EndArrowSize",2)
        geo=node(shape,"Section",{"N":"Geometry","IX":"0"})
        cell(geo,"NoFill",1)
        for j,(px,py) in enumerate(pts,1):
            row=node(geo,"Row",{"T":"MoveTo" if j==1 else "LineTo","IX":str(j)})
            dx,dy=px-sx,py-sy
            cell(row,"X",dx,f"Width*({dx/w:.16g})" if abs(w)>1e-12 else None)
            cell(row,"Y",dy,f"Height*({dy/h:.16g})" if abs(h)>1e-12 else None)
    elif kind == "text":
        stats["text_shapes"] += 1
        for n,v in {"LinePattern":0,"FillPattern":0,"VerticalAlign":1,
                    "LeftMargin":0,"RightMargin":0,"TopMargin":0,"BottomMargin":0}.items(): cell(shape,n,v)
        chars=node(shape,"Section",{"N":"Character"})
        paras=node(shape,"Section",{"N":"Paragraph"})
        para=node(paras,"Row",{"IX":"0"});cell(para,"HorzAlign",1)
        text_node=node(shape,"Text")
        for j,(text,position,italic) in enumerate(r["runs"]):
            row=node(chars,"Row",{"IX":str(j)})
            for n,v in {"Font":"Times New Roman","FontScale":1,"Size":r["size"]/72,"Color":r["color"],
                        "Style":int(r["bold"])+2*int(italic),"Pos":position}.items(): cell(row,n,v)
            cp=node(text_node,"cp",{"IX":str(j)}); cp.tail=text
    return shape


for r in sorted(records,key=lambda r:(r["z"],r["order"])):
    if r["parent"] is not None:
        continue
    if r in modules:
        children=r["children"]
        titles=["".join(x[0] for x in c["runs"]) for c in children if c["kind"]=="text"]
        title=titles[0] if titles else f"Module {r['order']}"
        group=new_shape(shapes_root,re.sub(r"[^A-Za-z0-9]+","_",title),"Group")
        r["id"]=int(group.get("ID"));stats["groups"]+=1
        transform(group,r["bounds"],(0,H))
        node(group,"Cell",{"N":"SelectMode","V":"0"})
        sub=node(group,"Shapes")
        x,y,w,h=r["bounds"]
        for c in sorted([r]+children,key=lambda c:(c["z"],c["order"])):
            draw(sub,c,(x,y+h),module_body=c is r)
    else:
        draw(shapes_root,r)

bindings=[]
for r in records:
    if r["kind"] != "arrow": continue
    for endpoint,p in (("BeginX",r["points"][0]),("EndX",r["points"][-1])):
        px,py=p
        candidates=[]
        for m in modules:
            x,y,w,h=m["bounds"]
            if x-3 <= px <= x+w+3 and y-3 <= py <= y+h+3:
                distance=min(abs(px-x),abs(px-x-w),abs(py-y),abs(py-y-h))
                if distance<=3: candidates.append((distance,w*h,m))
        if candidates:
            m=min(candidates,key=lambda item:item[:2])[2];x,y,w,h=m["bounds"]
            bindings.append({"line":r["id"],"cell":endpoint,"target":m["id"],
                             "x":max(0,min(1,(px-x)/w)),"y":max(0,min(1,1-(py-y)/h))})

document=ET.Element(f"{{{NS}}}VisioDocument",{"{http://www.w3.org/XML/1998/namespace}space":"preserve"})
node(document,"DocumentSettings",{"TopPage":"0","DefaultTextStyle":"0","DefaultLineStyle":"0","DefaultFillStyle":"0"})
faces=node(document,"FaceNames");node(faces,"FaceName",{"NameU":"Times New Roman"})
styles=node(document,"StyleSheets");default=node(styles,"StyleSheet",{"ID":"0","NameU":"Normal"})
for n,v in {"EnableLineProps":1,"EnableFillProps":1,"EnableTextProps":1,"LineWeight":1/72,
            "LineColor":"#21212a","LinePattern":1,"FillPattern":0,"VerticalAlign":1}.items():cell(default,n,v)
char=node(node(default,"Section",{"N":"Character"}),"Row",{"IX":"0"})
for n,v in {"Font":"Times New Roman","FontScale":1,"Size":12/72,"Color":"#21212a"}.items():cell(char,n,v)
node(document,"DocumentSheet",{"NameU":"TheDoc","LineStyle":"0","FillStyle":"0","TextStyle":"0"})

pages=ET.Element(f"{{{NS}}}Pages")
page=node(pages,"Page",{"ID":"0","NameU":"Mut-MAPPO","Name":"Mut-MAPPO","ViewScale":"0.65",
                        "ViewCenterX":str(W/SCALE/2),"ViewCenterY":str(H/SCALE/2)})
sheet=node(page,"PageSheet",{"LineStyle":"0","FillStyle":"0","TextStyle":"0"})
for n,v in {"PageWidth":W/SCALE,"PageHeight":H/SCALE,"PageScale":1,"DrawingScale":1,
            "DrawingSizeType":0,"DrawingScaleType":0,"DrawingResizeType":0,"PrintPageOrientation":2}.items():cell(sheet,n,v)
node(page,"Rel",{f"{{{REL}}}id":"rId1"})


def rels(entries):
    root=ET.Element(f"{{{PKG}}}Relationships")
    for ix,(kind,target) in enumerate(entries,1):
        ET.SubElement(root,f"{{{PKG}}}Relationship",{"Id":f"rId{ix}","Type":kind,"Target":target})
    return xml_bytes(root)


types=ET.Element(f"{{{CT}}}Types")
for extension,mime in (("rels","application/vnd.openxmlformats-package.relationships+xml"),("xml","application/xml")):
    ET.SubElement(types,f"{{{CT}}}Default",{"Extension":extension,"ContentType":mime})
for part,mime in (("/visio/document.xml","application/vnd.ms-visio.drawing.main+xml"),
                  ("/visio/pages/pages.xml","application/vnd.ms-visio.pages+xml"),
                  ("/visio/pages/page1.xml","application/vnd.ms-visio.page+xml")):
    ET.SubElement(types,f"{{{CT}}}Override",{"PartName":part,"ContentType":mime})

base="http://schemas.microsoft.com/visio/2010/relationships/"
reference=HERE/"visio_native_reference.vsdx"
if not reference.exists():
    raise FileNotFoundError("Create the clean native Visio container with probe_visio.ps1 first")
# Native Visio accepted each generated XML content part in isolation, but
# rejected the hand-written minimal OPC envelope even with an empty page.
# Preserve the clean, locally generated native envelope, including window and
# document-property relationships, while replacing all drawing content.
with ZipFile(reference) as package:
    parts={name:package.read(name) for name in package.namelist()}
parts.update({"visio/document.xml":xml_bytes(document),
              "visio/pages/pages.xml":xml_bytes(pages),
              "visio/pages/page1.xml":xml_bytes(page_root)})
draft=HERE/"mut_mappo_network_editable_draft.vsdx"
for name,value in parts.items():
    if name.endswith((".xml",".rels")) or name=="[Content_Types].xml": ET.fromstring(value)
with ZipFile(draft,"w",ZIP_DEFLATED) as package:
    for filename,value in parts.items():package.writestr(filename,value)
with ZipFile(draft) as package:
    assert package.testzip() is None
    assert not any("media/" in name for name in package.namelist())
stats.update(native_shape_count=next_id,glue_endpoints=len(bindings),draft=str(draft),status="DRAFT_XML_VALIDATED")
(HERE/"visio_bindings.json").write_text(json.dumps(bindings,indent=2),encoding="utf-8")
(HERE/"visio_draft_manifest.json").write_text(json.dumps(stats,indent=2),encoding="utf-8")
print(json.dumps(stats,ensure_ascii=False))
