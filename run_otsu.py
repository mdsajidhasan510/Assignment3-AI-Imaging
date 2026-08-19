import argparse,pandas as pd
from src.data import paired_paths,load_gray,load_mask
from src.classical import otsu_mask
from src.metrics import dice_score,iou_score
p=argparse.ArgumentParser(); p.add_argument('--data-dir',required=True); p.add_argument('--split',default='val'); p.add_argument('--output-dir',default='outputs/otsu'); a=p.parse_args(); rows=[]
for im,ma in paired_paths(a.data_dir,a.split):
 x=load_gray(im); y=load_mask(ma); z=otsu_mask(x); rows.append({'image_id':im.stem,'dice':dice_score(z,y),'iou':iou_score(z,y)})
df=pd.DataFrame(rows); print(df); print('Mean Dice:',df.dice.mean()); df.to_csv(a.output_dir+'/otsu_metrics.csv',index=False)
