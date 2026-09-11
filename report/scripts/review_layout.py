"""Render contact sheets for manual review; never marks a PDF as reviewed."""
from pathlib import Path
from PIL import Image, ImageDraw
import sys
root=Path(sys.argv[1])
pages=sorted(root.glob('page-*.png'))
for start in range(0,len(pages),6):
    sheet=Image.new('RGB',(3*660,2*880),'#dddddd')
    for offset,path in enumerate(pages[start:start+6]):
        im=Image.open(path).convert('RGB')
        im.thumbnail((650,850))
        x=(offset%3)*660;y=(offset//3)*880
        sheet.paste(im,(x,y+25))
        ImageDraw.Draw(sheet).text((x+10,y+5),path.stem,fill='black')
    sheet.save(root/f'sheet-{start//6+1}.png')
