import argparse,pandas as pd,torch
from src.data import paired_paths,load_gray,load_mask
from src.unet import UNet
from src.metrics import dice_score,iou_score
a=argparse.ArgumentParser(); a.add_argument('--data-dir',required=True); a.add_argument('--checkpoint',required=True); a.add_argument('--split',default='val'); a.add_argument('--output',default='outputs/unet_metrics.csv'); args=a.parse_args(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); c=torch.load(args.checkpoint,map_location=dev); m=UNet(width=c.get('width',32)).to(dev); m.load_state_dict(c['model_state']); m.eval(); rows=[]
for im,ma in paired_paths(args.data_dir,args.split):
 x=load_gray(im); y=load_mask(ma)
 with torch.no_grad(): z=(torch.sigmoid(m(torch.tensor(x[None,None]).float().to(dev))).cpu().numpy()[0,0]>.5).astype('uint8')
 rows.append({'image_id':im.stem,'dice':dice_score(z,y),'iou':iou_score(z,y)})
from pathlib import Path
df=pd.DataFrame(rows); print(df); print('Mean Dice:',df.dice.mean(),'Mean IoU:',df.iou.mean()); Path(args.output).parent.mkdir(parents=True,exist_ok=True); df.to_csv(args.output,index=False)
