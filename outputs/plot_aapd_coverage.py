from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json

root = Path(__file__).resolve().parent
out = root / 'aapd_analysis'
out.mkdir(exist_ok=True)
coverage = json.loads((root / 'benchmark/AAPD/aligned-v1/coverage_metrics.json').read_text())
local = [r['coverage'] * 100 for r in coverage['results']]
im = Image.new('RGB', (1400, 850), 'white')
d = ImageDraw.Draw(im)
def text(x,y,s,size=26,color='#253047',anchor=None):
    d.text((x,y),s,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size),fill=color,anchor=anchor)
text(90,35,'AAPD',38)
for x,color,label in [(930,'#8b9aad','论文'),(1140,'#2878bc','本次')]:
    d.line((x,60,x+48,60),fill=color,width=5)
    d.ellipse((x+18,54,x+30,66),fill=color)
    text(x+60,42,label)
left,right,top,bottom=155,1190,160,660
text(90,105,'覆盖率 (%)',27)
def point(x,y): return (left+x/10*(right-left),bottom-y/100*(bottom-top))
for tick in range(0,101,20):
    y=point(0,tick)[1]
    d.line((left,y,right,y),fill='#e2e7ed',width=2)
    text(left-25,y,str(tick),24,'#596578','rm')
for tick in range(11):
    x=point(tick,0)[0]
    text(x,bottom+23,str(tick),24,'#596578','mt')
text((left+right)/2,725,'迭代次数',27,anchor='mt')
for values,color in [([44.90,77.56],'#8b9aad'),(local,'#2878bc')]:
    a,b=point(0,values[0]),point(10,values[1])
    for k in range(0,100,2):
        t,u=k/100,min((k+1)/100,1)
        d.line((a[0]+(b[0]-a[0])*t,a[1]+(b[1]-a[1])*t,a[0]+(b[0]-a[0])*u,a[1]+(b[1]-a[1])*u),fill=color,width=4)
    for idx,(x,y) in enumerate([a,b]):
        d.ellipse((x-7,y-7,x+7,y+7),fill=color)
        if idx==0:
            text(x+16,y-13,f'{values[idx]:.2f}',27)
        else:
            offset=-33 if color=='#2878bc' else 24
            text(x+19,y+offset,f'{values[idx]:.2f}',27)
text(90,792,'仅标出初始与第 10 轮；虚线连接不代表逐轮实测值。本次为校正前数据。',23,'#596578')
im.save(out/'aapd-coverage-endpoints.png',dpi=(200,200))
