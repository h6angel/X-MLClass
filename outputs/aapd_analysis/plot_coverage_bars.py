from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

p=Path(__file__).resolve().parent
im=Image.new('RGB',(1400,850),'white')
d=ImageDraw.Draw(im)
def text(x,y,s,size=28,color='#243247',anchor=None):
    d.text((x,y),s,font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',size),fill=color,anchor=anchor)
blue,orange='#1464c0','#e16b19'
text(90,40,'AAPD',40)
for x,c,label in [(950,blue,'论文'),(1140,orange,'本次')]:
    d.rectangle((x,52,x+25,77),fill=c)
    text(x+40,44,label)
text(90,115,'覆盖率 (%)',27)
left,right,top,bottom=145,1290,190,675
for tick in range(0,101,20):
    y=bottom-tick/100*(bottom-top)
    d.line((left,y,right,y),fill='#e7ebef',width=2)
    text(left-24,y,str(tick),25,'#687586','rm')
for center,label,paper,local in [(450,'初始覆盖率',44.90,73.58),(1000,'最终覆盖率',77.56,79.25)]:
    for x,val,c in [(center-130,paper,blue),(center+20,local,orange)]:
        y=bottom-val/100*(bottom-top)
        d.rectangle((x,y,x+110,bottom),fill=c)
        text(x+55,y-18,f'{val:.2f}%',30,anchor='mb')
    text(center,bottom+30,label,31,anchor='mt')
text(90,795,'本次为真实标签解析校正前数据。',23,'#687586')
im.save(p/'aapd-coverage-bars.png',dpi=(240,240))
