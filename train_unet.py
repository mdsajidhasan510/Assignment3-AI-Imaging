import argparse,random,numpy as np,torch
from torch.utils.data import Dataset,DataLoader
from pathlib import Path
from src.data import paired_paths,load_gray,load_mask
from src.unet import UNet,SoftDiceLoss
from src.metrics import dice_score
class DS(Dataset):
 def __init__(self,pairs,aug=False): self.pairs,self.aug=pairs,aug
 def __len__(self): return len(self.pairs)
 def __getitem__(self,i):
  p,m=self.pairs[i]; x=load_gray(p); y=load_mask(m)
  if self.aug:
   k=random.randrange(4); x=np.rot90(x,k).copy(); y=np.rot90(y,k).copy()
   if random.random()<.5:x=np.fliplr(x).copy(); y=np.fliplr(y).copy()
  return torch.tensor(x[None]).float(),torch.tensor(y[None]).float()
def eval_model(model,loader,dev):
 model.eval(); s=[]
 with torch.no_grad():
  for x,y in loader:
   z=(torch.sigmoid(model(x.to(dev)))>.5).cpu().numpy(); t=y.numpy(); s += [dice_score(a,b) for a,b in zip(z,t)]
 return float(np.mean(s))
a=argparse.ArgumentParser(); a.add_argument('--data-dir',required=True); a.add_argument('--epochs',type=int,default=12); a.add_argument('--batch-size',type=int,default=4); a.add_argument('--lr',type=float,default=5e-4); a.add_argument('--width',type=int,default=32); a.add_argument('--checkpoint',default='models/unet_dice.pt'); args=a.parse_args()
torch.manual_seed(42); np.random.seed(42); random.seed(42); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); tr=DataLoader(DS(paired_paths(args.data_dir,'train'),True),batch_size=args.batch_size,shuffle=True); va=DataLoader(DS(paired_paths(args.data_dir,'val')),batch_size=args.batch_size); model=UNet(width=args.width).to(dev); loss=SoftDiceLoss(); opt=torch.optim.Adam(model.parameters(),lr=args.lr); best=-1; Path(args.checkpoint).parent.mkdir(parents=True,exist_ok=True)
for e in range(args.epochs):
 model.train(); ls=[]
 for x,y in tr:
  opt.zero_grad(); l=loss(model(x.to(dev)),y.to(dev)); l.backward(); opt.step(); ls.append(l.item())
 d=eval_model(model,va,dev); print(f'Epoch {e+1}/{args.epochs} loss={np.mean(ls):.4f} val_dice={d:.4f}')
 if d>best: best=d; torch.save({'model_state':model.state_dict(),'width':args.width,'val_dice':d},args.checkpoint)
print('Best validation Dice:',best)
