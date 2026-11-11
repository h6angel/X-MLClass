from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parent / 'aapd_analysis'
OUT.mkdir(exist_ok=True)
im = Image.new('RGB', (1600, 780), '#ffffff')
d = ImageDraw.Draw(im)
font_path = 'C:/Windows/Fonts/msyh.ttc'
def font(size):
    return ImageFont.truetype(font_path, size)
def text(x, y, value, size=25, fill='#253047', anchor=None):
    d.text((x,y), value, font=font(size), fill=fill, anchor=anchor)

text(75, 35, 'AAPD', 38)
colors = ['#8b9aad', '#2878bc']
for x, color, label in [(1160, colors[0], '论文'), (1350, colors[1], '本次')]:
    d.rectangle((x, 45, x+25, 70), fill=color)
    text(x+38, 40, label, 25)

models = ['DeBERTa', 'BART', 'XLM-R']
panels = [('P@1', [35.73,33.23,27.13], [36.94,32.23,29.03]),
          ('P@3', [40.09,37.35,34.67], [42.31,36.34,37.89])]
for idx,(metric,paper,local) in enumerate(panels):
    left = 100 + idx*780
    right = left+620
    top, bottom = 180, 655
    text(left, 115, metric + ' (%)', 30)
    for tick in range(0,51,10):
        y=bottom-tick/50*(bottom-top)
        d.line((left,y,right,y),fill='#e2e7ed',width=2)
        text(left-22,y,str(tick),23,'#596578',anchor='rm')
    for j,model in enumerate(models):
        center=left+110+j*205
        for x,value,color in [(center-73,paper[j],colors[0]),(center+7,local[j],colors[1])]:
            y=bottom-value/50*(bottom-top)
            d.rectangle((x,y,x+65,bottom),fill=color)
            text(x+32,y-18,f'{value:.2f}',25,anchor='mb')
        text(center,bottom+25,model,27,anchor='mt')
im.save(OUT/'aapd-comparison-simple.png',dpi=(200,200))
print(OUT/'aapd-comparison-simple.png')
