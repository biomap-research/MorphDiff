import argparse, os, sys, glob
sys.path.append("..")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler
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


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    print(config.model)
    model = instantiate_from_config(config.model)
    print(model)
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

def validate_n_samples(value):
    ivalue = int(value)
    if ivalue % 4 != 0:
        raise argparse.ArgumentTypeError(f"{value} is not divisible by 4")
    return ivalue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--gene_path",
        type=str,
        nargs="?",
        default="/data/wangxuesong/CPA/different_dose_10_data.npy",
        help="the prompt to render"
    )
    
    parser.add_argument(
        "--index_path",
        type=str,
        nargs="?",
        default=None,
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="./outputs/different_dose_10_data"
    )

    parser.add_argument(
        "--output_name",
        type=str,
        nargs="?",
        help="name for 5c generated",
        default="./outputs/different_dose_10_data"
    )

    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--dpm_solver",
        action='store_true',
        help="use dpm_solver sampling",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
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
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    '''parser.add_argument(
        "--n_samples",
        type=validate_n_samples,
        default=100,
        help="how many samples to produce (must be divisible by 4)") '''

    parser.add_argument(
        "--scale",
        type=float,
        default=5.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=80,
        help="batch_size",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default='/data2/wangxuesong/stable-diffusion4morphology/configs/latent-diffusion/morphology_single_channel.yaml',
        help="config file path",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='/data2/wangxuesong/stable-diffusion4morphology/logs/morphology_single_channel/2023-10-30T22-52-01_morphology_single_channel/checkpoints/last.ckpt',
        help="model file path",
    )
    opt = parser.parse_args()


    config = OmegaConf.load(opt.config_path)  # TODO: Optionally download from same location as ckpt and chnage this logic
    model = load_model_from_config(config, opt.model_path)  # TODO: check path

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.dpm_solver:
        sampler = DPMSolverSampler(model)
    elif opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    gene_all = np.load(opt.gene_path)
    if opt.index_path is not None:
        index=np.load(opt.index_path)
        gene_all=gene_all[index]


    #sample_path = os.path.join(outpath, "samples")
    #os.makedirs(sample_path, exist_ok=True)
    #base_count = len(os.listdir(sample_path))

    image_of_5c=[]
    number_of_sample=int(gene_all.shape[0]/opt.batch_size)*opt.batch_size
    for i in range(number_of_sample // opt.batch_size):
        gene = gene_all[opt.batch_size * i : opt.batch_size * (i+1), :]
        all_samples=list()
        with torch.no_grad():
            with model.ema_scope():
                uc = None
                if opt.scale != 1.0:
                    uc = model.get_learned_conditioning(4 * [""])
                for n in trange(opt.n_iter, desc="Sampling"):
                    c = model.get_learned_conditioning(gene)
                    #shape = [4, opt.H//8, opt.W//8]
                    shape = [4, opt.H//4, opt.W//4]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.batch_size,
                                                    shape=shape,
                                                    verbose=False,
                                                    unconditional_guidance_scale=opt.scale,
                                                    unconditional_conditioning=uc,
                                                    eta=opt.ddim_eta)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
                    #print(x_samples_ddim.shape)

                    for x_sample in x_samples_ddim:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        #print(x_sample.shape)
                        #print(x_sample)
                        #input('pause')
                        image_of_5c.append(x_sample.astype(np.uint8))
                        #Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.png"))
                        #base_count += 1
                    #all_samples.append(x_samples_ddim)


        # additionally, save as grid
        '''grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=opt.n_samples)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'batch_{i}.png'))'''
    image_of_5c=np.stack(image_of_5c)
    np.save(opt.output_name,image_of_5c)
    for idx,t in enumerate(image_of_5c):
    #print(idx)
        expanded_array = np.expand_dims(t[:,:,1], axis=2)
        result0 = np.repeat(expanded_array, 3, axis=2)

        expanded_array = np.expand_dims(t[:,:,2], axis=2)
        result1 = np.repeat(expanded_array, 3, axis=2)

        expanded_array = np.expand_dims(t[:,:,0], axis=2)
        result2 = np.repeat(expanded_array, 3, axis=2)

        expanded_array = np.expand_dims(t[:,:,4], axis=2)
        result3 = np.repeat(expanded_array, 3, axis=2)

        expanded_array = np.expand_dims(t[:,:,3], axis=2)
        result4 = np.repeat(expanded_array, 3, axis=2)

        composite(result0,result1,result2,result3,result4,opt.outdir,str(idx)+'.png')
    print(f"Your samples are ready and waiting four you here: \n{outpath} \nEnjoy.")