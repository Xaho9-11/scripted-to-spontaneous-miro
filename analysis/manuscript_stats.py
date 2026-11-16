from pathlib import Path
from math import comb, sqrt
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[1]
BOOK = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "phase2_data.xlsx"  # not published
OUT = ROOT / "figures"; OUT.mkdir(exist_ok=True)
df = pd.read_excel(BOOK, sheet_name="Data Entry", header=2)
df = df[pd.to_numeric(df["#"], errors="coerce").notna()].reset_index(drop=True)

def nums(cols):
    return df.iloc[:, cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
def alpha(x):
    k=x.shape[1]
    return k/(k-1)*(1-x.var(axis=0,ddof=1).sum()/x.sum(axis=1).var(ddof=1))
def paired(name,a,b):
    d=a-b; se=d.std(ddof=1)/sqrt(len(d)); t=d.mean()/se; dz=d.mean()/d.std(ddof=1)
    ci95=(d.mean()-2.119905299*se,d.mean()+2.119905299*se)
    ci90=(d.mean()-1.745883676*se,d.mean()+1.745883676*se)
    print(name,"S",a.mean(),a.std(ddof=1),"M",b.mean(),b.std(ddof=1),"diff",d.mean(),"t",t,"dz",dz,"CI95",ci95,"CI90",ci90)
    return d,ci90

ecl_si,ecl_mi=nums([8,9,10]),nums([11,12,13])
gs_si,gs_mi=nums([16,17,18,19,20,21]),nums([22,23,24,25,26,27])
ros_si,ros_mi=nums([30,31,32,33,34,35]),nums([36,37,38,39,40,41])
ecl_s,ecl_m=ecl_si.mean(1),ecl_mi.mean(1)
gs_s,gs_m=gs_si.mean(1),gs_mi.mean(1)
ros_s,ros_m=ros_si.mean(1),ros_mi.mean(1)
dg,cig=paired("GS",gs_s,gs_m); de,cie=paired("ECL",ecl_s,ecl_m); dr,cir=paired("RoSAS",ros_s,ros_m)
print("alphas",{"ECL-S":alpha(ecl_si),"ECL-M":alpha(ecl_mi),"GS-S":alpha(gs_si),"GS-M":alpha(gs_mi),"RoSAS-S":alpha(ros_si),"RoSAS-M":alpha(ros_mi)})
pos,neg,ties=int((dg>0).sum()),int((dg<0).sum()),int((dg==0).sum())
p_sign=2*sum(comb(pos+neg,k) for k in range(0,min(pos,neg)+1))/(2**(pos+neg))
print("sign",pos,neg,ties,p_sign,"orders",df["Order"].value_counts().to_dict(),"profiles",df["Profile"].value_counts().to_dict())

FONT=Path("C:/Windows/Fonts/arial.ttf"); BOLD=Path("C:/Windows/Fonts/arialbd.ttf")
def font(size,bold=False): return ImageFont.truetype(str(BOLD if bold else FONT),size)
def text(draw,xy,s,size=28,fill="#222",anchor=None,bold=False): draw.text(xy,s,font=font(size,bold),fill=fill,anchor=anchor)

# Paired animacy plot
W,H=1260,820; im=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(im)
L,R,T,B=120,1160,130,690; x1,x2=300,980
d.line((L,T,L,B),fill="#555",width=3); d.line((L,B,R,B),fill="#555",width=3)
def y(v): return B-(v-1)/6*(B-T)
for v in range(1,8):
    yy=y(v); d.line((L-10,yy,L,yy),fill="#555",width=2); text(d,(L-22,yy),str(v),24,anchor="rm")
colors={"AP":"#4C72B0","WC":"#DD8452","PB":"#C84F55"}
for i,row in df.iterrows():
    c=colors[row["Profile"]]; d.line((x1,y(gs_s[i]),x2,y(gs_m[i])),fill=c,width=3)
    d.ellipse((x1-6,y(gs_s[i])-6,x1+6,y(gs_s[i])+6),fill=c); d.ellipse((x2-6,y(gs_m[i])-6,x2+6,y(gs_m[i])+6),fill=c)
d.line((x1,y(gs_s.mean()),x2,y(gs_m.mean())),fill="black",width=8)
for xx,v in [(x1,gs_s.mean()),(x2,gs_m.mean())]: d.polygon([(xx, y(v)-13),(xx+13,y(v)),(xx,y(v)+13),(xx-13,y(v))],fill="black")
text(d,(x1,B+34),"Spontaneous",30,anchor="ma"); text(d,(x2,B+34),"Scripted",30,anchor="ma")
# Draw the y-axis title vertically so it remains readable at column width.
axis_label = "Godspeed Animacy (mean, 1\u20137)"
label_box = font(28).getbbox(axis_label)
label_im = Image.new("RGBA", (label_box[2] + 12, label_box[3] - label_box[1] + 12), (255, 255, 255, 0))
label_draw = ImageDraw.Draw(label_im)
label_draw.text((6, 6 - label_box[1]), axis_label, font=font(28), fill="#222")
label_im = label_im.rotate(90, expand=True)
im.paste(label_im, (18, ((T + B) - label_im.height) // 2), label_im)
# Horizontal legend in the reserved band above the plotting area.
legend=[("AP","Ambient Presence"),("WC","Watchful Companion"),("PB","Pattern Breaker")]
legend_x = [150, 435, 745]
for xx,(k,label) in zip(legend_x,legend):
    yy=62; d.line((xx,yy,xx+38,yy),fill=colors[k],width=4); d.ellipse((xx+13,yy-6,xx+25,yy+6),fill=colors[k]); text(d,(xx+50,yy),label,22,anchor="lm")
d.line((1010,62,1048,62),fill="black",width=7); text(d,(1060,62),"Group mean",22,anchor="lm")
text(d,(L+15,B-50),"Paired t(16) = 2.99, p = .009, d_z = 0.73",23)
im.save(OUT/"Individual Paired Scores.png")

# Non-inferiority plot using two-sided 90% CIs (= one-sided 95% bounds)
W,H=1260,560; im=Image.new("RGB",(W,H),"white"); d=ImageDraw.Draw(im)
L,R,T,B=260,1160,75,440; xmin,xmax=-1.7,1.35
def x(v): return L+(v-xmin)/(xmax-xmin)*(R-L)
d.line((L,B,R,B),fill="#555",width=3)
for v in [-1.5,-1.0,-.5,0,.5,1.0]:
    xx=x(v); d.line((xx,B,xx,B+9),fill="#555",width=2); text(d,(xx,B+18),f"{v:.1f}",22,anchor="ma")
d.line((x(0),T,x(0),B),fill="#999",width=2); d.line((x(.5),T,x(.5),B),fill="#D9534F",width=4)
text(d,(x(.5)+12,T),"Non-inferiority margin \u0394 = 0.5",23,fill="#D9534F")
for yy,name,diff,ci in [(200,"Klepsch ECL",de.mean(),cie),(335,"RoSAS Discomfort",dr.mean(),cir)]:
    text(d,(L-28,yy),name,26,anchor="rm"); d.line((x(ci[0]),yy,x(ci[1]),yy),fill="#2E8B57",width=8)
    d.ellipse((x(diff)-13,yy-13,x(diff)+13,yy+13),fill="#2E8B57")
    text(d,(x(ci[1])+20,yy),f"D = {diff:.2f}, 90% CI [{ci[0]:.2f}, {ci[1]:.2f}]",22,anchor="lm")
text(d,((L+R)//2,H-45),"Mean difference (Spontaneous \u2212 Scripted)",26,anchor="ma")
im.save(OUT/"Non-inferiority Test.png")
