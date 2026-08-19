import argparse,torch,numpy as np
from src.data import load_gray
from src.unet import UNet
from src.pipeline import run_test
a=argparse.ArgumentParser(); a.add_argument('--data-dir',required=True); a.add_argument('--checkpoint',required=True); a.add_argument('--output-dir',default='outputs/final'); a.add_argument('--text-model',default='llama3.1:8b'); args=a.parse_args(); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); c=torch.load(args.checkpoint,map_location=dev); m=UNet(width=c.get('width',32)).to(dev); m.load_state_dict(c['model_state']); m.eval()
def predict(x):
 with torch.no_grad(): p=torch.sigmoid(m(torch.tensor(x[None,None]).float().to(dev))).cpu().numpy()[0,0]
 return (p>.5).astype('uint8')
print(run_test(args.data_dir,m,args.output_dir,args.text_model,predict).to_string(index=False))
