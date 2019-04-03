# workaround for fastai crash on Mac: https://github.com/scikit-optimize/scikit-optimize/issues/637
import matplotlib
matplotlib.use('PS')

from fastai.text import *
from fastai.callbacks.tracker import *
from fastai.distributed import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default='~/data/wikitext-2-raw/')
parser.add_argument('--save', type=str, default='first_run')
parser.add_argument('--load', type=str, default=None)
parser.add_argument("--local_rank", type=int)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--mem_len", type=int, default=512)
parser.add_argument("--bptt", type=int, default=512)
parser.add_argument('--half', action='store_true', help='Use half precision')
parser.add_argument('--wd', type=float, default=1e-3, help='weight decay for adam')
parser.add_argument('--epochs', type=int, default=5, help='num epochs')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()

args.path = args.path.replace('~', os.environ['HOME'])

if args.local_rank != 0:
    f = open('/dev/null', 'w')
    sys.stdout = f

if torch.cuda.device_count()>0:
  torch.cuda.set_device(args.local_rank)
else:
  print("Not using GPU")

running_on_mac = False  # distributed is missing on Mac, keep track
try:
  torch.distributed.init_process_group(backend='nccl', init_method='env://')
except:
  print("Couldn't init process group...running on mac?")
  running_on_mac = True

bs=args.batch_size
bptt=args.bptt
path = Path(args.path)
data = text_data = load_data(path, bs=bs, bptt=bptt)

vocab = data.train_ds.vocab
vocab_size = len(vocab.itos)

tfmerXL_lm_config['ctx_len'] = 512
tfmerXL_lm_config['mem_len'] = args.mem_len

full_clip = None if args.half else 0.25
learn = language_model_learner(data, TransformerXL, clip=full_clip)

if args.load:
    load_path = Path(args.path)/args.load
    state = torch.load(load_path, map_location='cpu')
    get_model(learn.model).load_state_dict(state['model'], strict=False)
    learn.model.cuda()
if args.save:
    save_path = Path(args.path)/learn.model_dir/args.save
    save_path.parent.mkdir(parents=True, exist_ok=True)
if args.half: learn = learn.to_fp16(clip=0.25, dynamic=True)

if not running_on_mac:
  learn = learn.to_distributed(args.local_rank)
  
if args.local_rank == 0: learn.callbacks.append(SaveModelCallback(learn, name=f'{args.save}_best'))
learn.callbacks.append(EarlyStoppingCallback(learn))

learn.fit_one_cycle(args.epochs, args.lr, div_factor=25, moms=(0.7,0.5))

if args.local_rank == 0: learn.save(f'{args.save}')
