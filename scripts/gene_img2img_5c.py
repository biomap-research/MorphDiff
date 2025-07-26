"""make variations of input image"""

import argparse, os, sys, glob
sys.path.append('/code/MorphDiff/')
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
import time
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from sys import getsizeof as getsize

import cv2

def composite(tmp1,tmp2,tmp3,tmp4,tmp5,output_path,output_name):    #ERSyto ERSytoBleed Hoechst Mito Ph_golgi
        #colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0.4,0.4,0), (0.2, 0.4, 0.2)]  #orf
        colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0.4, 0.4), (0.4, 0.2, 0.2)]  #  drug
        #a = x[:,0,].unsqueeze(1)
        #tmp1=tmp1[np.newaxis,:]
        temp=[tmp1,tmp2,tmp3,tmp4,tmp5]
        image = np.zeros(temp[0].shape)
        for img in range(len(temp)):
                image += temp[img] * colors[img]
        #try:
        image[:,:,0] = (image[:,:,0]-np.min(image[:,:,0])) / (np.max(image[:,:,0])-np.min(image[:,:,0])) * 256
        image[:,:,1] = (image[:,:,1]-np.min(image[:,:,1])) / (np.max(image[:,:,1])-np.min(image[:,:,1])) * 256
        image[:,:,2] = (image[:,:,2]-np.min(image[:,:,2])) / (np.max(image[:,:,2])-np.min(image[:,:,2])) * 256
                #return image
        cv2.imwrite( output_path+output_name,image)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def load_img(image):
    #image = Image.open(path).convert("RGB")
    #w, h = image.size
    w=image.shape[0]
    h=image.shape[1]
    #print(f"loaded input image of size ({w}, {h}) from {path}")
    #w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    #image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.*image - 1.


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    
    parser.add_argument(
        "--gene_path",
        type=str,
        nargs="?",
        default="/data2/wangxuesong/morphdata/orf_segment_new/train_test_ood/test_gene_count.npy",
        help="the prompt to render"
    )

    parser.add_argument(
        "--output_name",
        type=str,
        nargs="?",
        default="./outputs/drug_ood_sample_12_14_5c.npy",
        help="the prompt to render"
    )

    parser.add_argument(
        "--init_img_path",
        type=str,
        default="/data2/wangxuesong/morphdata/orf_segment_new/train_test_ood/control_img4test",
        nargs="?",
        help="path to the input image"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2img-samples"
    )

    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )

    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save indiviual samples. For speed measurements.",
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across all samples ",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=16,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--strength",
        type=float,
        default=0.75,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )

    opt = parser.parse_args()
    seed_everything(opt.seed)

    #config = OmegaConf.load(f"{opt.config}")
    #model = load_model_from_config(config, f"{opt.ckpt}")
    
    config = OmegaConf.load(opt.config_path)  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, opt.model_path)  # TODO: 

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

  #  sample_path = os.path.join(outpath, "samples")
  #  os.makedirs(sample_path, exist_ok=True)
   # base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1
    
    gene_all = np.load(opt.gene_path)

    #assert os.path.isfile(opt.init_img)
    #init_image_order=[]
    init_image_list=[]
    #with open(opt.img_order_path, 'r') as file:
    #    img_order = [line.strip() for line in file]
    init_image_array=np.load(opt.init_img_path)
    #count=0
    #for i in init_image_array:
    #train_set=np.load(opt.train_set_path)
    for i in init_image_array:
        #print(count)
        init_image = load_img(i).to(device)
        #init_image = load_img(train_set[int(i)]).to(device)
        #print(getsize(init_image))
        #count=count+1
        
        #init_image = load_img(opt.init_img_path+'/'+i).to(device)
        init_image_list.append(repeat(init_image, '1 ... -> b ...', b=batch_size))
        #init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    
    
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")
    image_of_5c=[]
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    #for prompts in tqdm(data, desc="data"):
                    for idx,gene_counts in enumerate(tqdm(gene_all, desc="data")):
                        uc = None
                        if opt.scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        #if isinstance(prompts, tuple):
                        #    prompts = list(prompts)
                        c = model.get_learned_conditioning(gene_counts)
                        init_latent=model.get_first_stage_encoding(model.encode_first_stage(init_image_list[idx]))
                        # encode (scaled latent)
                        z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        # decode it
                        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,)

                        x_samples = model.decode_first_stage(samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                image_of_5c.append(x_sample.astype(np.uint8))


                toc = time.time()
    image_of_5c=np.stack(image_of_5c)
    np.save(opt.output_name,image_of_5c)

    for idx,t in enumerate(image_of_5c):
    #print(idx)
        expanded_array = np.expand_dims(t[:,:,0], axis=2)
        result0 = np.repeat(expanded_array, 3, axis=2)

        expanded_array = np.expand_dims(t[:,:,1], axis=2)
        result1 = np.repeat(expanded_array, 3, axis=2)

        expanded_array = np.expand_dims(t[:,:,2], axis=2)
        result2 = np.repeat(expanded_array, 3, axis=2)

        expanded_array = np.expand_dims(t[:,:,3], axis=2)
        result3 = np.repeat(expanded_array, 3, axis=2)

        expanded_array = np.expand_dims(t[:,:,4], axis=2)
        result4 = np.repeat(expanded_array, 3, axis=2)

        composite(result0,result1,result2,result3,result4,opt.outdir,str(idx)+'.png')
    print(opt.outdir)
    print(opt.output_name)
    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
