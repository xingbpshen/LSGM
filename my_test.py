from nvae import NVAE
from types import SimpleNamespace
from util import utils
import torch
from torch.multiprocessing import Process


def main(args):
    arch_instance_nvae = utils.get_arch_cells(args.arch_instance, args.use_se)
    vae = NVAE(args, arch_instance_nvae)
    vae = vae.cuda()

    # Generate random input same as CIFAR10
    x = torch.randn(1, 3, 32, 32).cuda()
    conditioning_inputs = {
        'gender': torch.tensor([0]).cuda(),
        'hair_color': torch.tensor([0]).cuda()
        # Add other conditioning variables as needed
    }
    print(vae(x, conditioning_inputs)[0].shape)
    print(vae.sample(1, 1, enable_autocast=True, conditioning_inputs=conditioning_inputs).shape)


args = SimpleNamespace(
    dataset="cifar10",
    num_channels_enc=128,
    num_channels_dec=128,
    num_postprocess_cells=2,
    num_preprocess_cells=2,
    num_latent_scales=1,
    num_cell_per_cond_enc=2,
    num_cell_per_cond_dec=2,
    num_preprocess_blocks=1,
    num_postprocess_blocks=1,
    num_latent_per_group=9,
    num_groups_per_scale=20,
    epochs=600,
    batch_size=32,
    weight_decay_norm=1e-2,
    num_nf=0,
    kl_anneal_portion=0.5,
    kl_max_coeff=1.0,
    channel_mult=[1, 2],
    seed=1,
    arch_instance='res_bnswish',
    num_process_per_node=1,
    use_se=True,
    log_sig_q_scale=5.0,
    num_x_bits=8,
    decoder_dist='dml',
    progressive_input_vae='none',
    node_rank=0,
    master_address='127.0.0.1',
    local_rank=0,
    global_rank=0,
    conditioning_vars=['gender', 'hair_color'],
    conditioning_dims={'gender': 2, 'hair_color': 5},
    embedding_dim=16,
)

size = args.num_process_per_node

if size > 1:
    args.distributed = True
    processes = []
    for rank in range(size):
        args.local_rank = rank
        global_rank = rank + args.node_rank * args.num_process_per_node
        global_size = args.num_proc_node * args.num_process_per_node
        args.global_rank = global_rank
        print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
        p = Process(target=utils.init_processes, args=(global_rank, global_size, main, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
else:
    # for debugging
    print('starting in debug mode')
    args.distributed = True
    utils.init_processes(0, size, main, args)


