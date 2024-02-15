""" Implement Gradient-Free Structured Pruning with Unlabeled Data Algorithm """

import torch
from utils.arch import get_ffn2, get_mha_proj
from efficiency.mac import compute_mac, mac_per_head, mac_per_neuron

def hijack(module, _list, _hijack_input, _stop_forward=False):
    # if _stop_forward=True, then it raise error after forwarding the module
    """
    Register a forward hook on a PyTorch module to capture input or output tensors during forward passes.

    * Inputs:
        - module (torch.nn.Module): The PyTorch module on which the forward hook will be registered.
        - _list (list): A list to which the modified data (input or output tensors) will be appended.
        - _hijack_input (bool): If True, the hook is applied to the input of the module; if False, to the output.
        - _stop_forward (bool, optional): If True, raise a StopForwardException after forwarding the module. Default is False.

    * Outputs:
        - handle (torch.utils.hooks.RemovableHandle): A handle to the registered forward hook. Can be used to unregister the hook later.

    * Raises:
        - StopForwardException: If _stop_forward is True, an exception is raised after forwarding the module.

    * Example:
        handles = []
        handles.append(hijack(output_proj, _inputs[sl], _hijack_input=True, _stop_forward=False))
        handles.append(apply_mask(output_proj, layer_mask))
    """
    if _hijack_input:
        def input_hook(_, inputs, __):
            _list.append(inputs[0].clone().data)
            if _stop_forward:
                raise StopFowardException

        handle = module.register_forward_hook(input_hook)
    else:
        def output_hook(_, __, outputs):
            _list.append(outputs.clone().data)
            if _stop_forward:
                raise StopFowardException

        handle = module.register_forward_hook(output_hook)
    return handle

@torch.no_grad()
def search_mac_kcm(
    config,
    head_importance,
    neuron_importance,
    seq_len,
    mac_constraint,
    log=False,
):
    assert mac_constraint < 1

    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    intermediate_size = config.intermediate_size
    hidden_size = config.hidden_size
    attention_head_size = int(hidden_size / num_attention_heads)

    original_mac = compute_mac(
        [num_attention_heads] * num_hidden_layers,
        [intermediate_size] * num_hidden_layers,
        seq_len,
        hidden_size,
        attention_head_size,
    )
    max_mac = mac_constraint * original_mac

    # neuron_importance = compute_fisher_info(neuron_grads) # 12(layers) * 3072(ffn dim)

    # Globally rank heads and neurons
    sorted_head_importance, sorted_head_indicies = head_importance.view(-1).sort(descending=True)
    sorted_neuron_importance, sorted_neuron_indicies = neuron_importance.view(-1).sort(descending=True)

    # Prune heads and neurons
    max_importance = 0
    for num_heads in range(1, num_hidden_layers * num_attention_heads + 1):
        heads_mac = mac_per_head(seq_len, hidden_size, attention_head_size) * num_heads
        neurons_mac = max_mac - heads_mac
        num_neurons = int(neurons_mac / mac_per_neuron(seq_len, hidden_size))
        num_neurons = max(num_neurons, 0)

        total_importance = sorted_head_importance[:num_heads].sum() + sorted_neuron_importance[:num_neurons].sum()
        if total_importance > max_importance:
            max_importance = total_importance
            head_indicies = sorted_head_indicies[:num_heads]
            neuron_indicies = sorted_neuron_indicies[:num_neurons]

    head_mask = torch.zeros(num_hidden_layers * num_attention_heads).cuda()
    head_mask[head_indicies] = 1.0
    head_mask = head_mask.view(num_hidden_layers, num_attention_heads)

    # only prune neuron
    heads_mac = mac_per_head(seq_len, hidden_size, attention_head_size) * num_attention_heads * num_hidden_layers
    neurons_mac = max_mac - heads_mac
    num_neurons = int(neurons_mac / mac_per_neuron(seq_len, hidden_size))
    neuron_indicies = sorted_neuron_indicies[:num_neurons]

    neuron_mask = torch.zeros(num_hidden_layers * intermediate_size).cuda()
    neuron_mask[neuron_indicies] = 1.0
    neuron_mask = neuron_mask.view(num_hidden_layers, intermediate_size)

    if log:
        for i in range(num_hidden_layers):
            # count how many zeros in head_mask[i]
            pruned_heads = (head_mask[i] == 0).nonzero()
            pruned_neurons = (neuron_mask[i] == 0).nonzero()
            if pruned_heads.size(0) > 0:
                print("layer {} prune {} heads: {}".format(i, pruned_heads.size(0), pruned_heads.squeeze().tolist()), end="")
            else:
                print("layer {} prune 0 heads".format(i), end="")
            print(", {} neurons".format(pruned_neurons.size(0)))

    return head_mask, neuron_mask

def d2_importance(model, type, structure, dataloader, log=False):
    """
    a function to compute d2 importance based on ffn activated output of neurons

    * Inputs
        - model: the target model to compress
        - type: type of normalization, choices = ["z-score", "log", "l2", "maxmin"]
        - structure: type of structure, choices = ["N" means neurons, "H" means attention heads]
        - dataloader: dataloader for unlabeled data
    * Outputs
        - importance: importance of neurons (layers x ffn dimension) or of heads (layers x attention head)
    """
    num_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    ffn_dim = model.config.intermediate_size
    attention_head_size = int(model.config.hidden_size / num_attention_heads)
    hidden_size = model.config.hidden_size
    # neuron_importance = torch.zeros(num_layers, ffn_dim).cuda()
    if structure == "N":
        importance = torch.zeros(num_layers, ffn_dim).cuda()
    else:
        importance = torch.zeros(num_layers, hidden_size).cuda() # L * d
        head_importance = torch.zeros(num_layers, num_attention_heads).cuda() # L * H

    # Register hook function
    model.eval()
    _inputs = {}
    handles = []
    for idx in range(num_layers):
        if structure == "N":
            output_proj = get_ffn2(model, idx).dense
        else:
            output_proj = get_mha_proj(model, idx).dense
        _inputs[idx] = []
        handles.append(
            hijack(output_proj, _inputs[idx], _hijack_input=True,
                   _stop_forward=False)
        )
    
    # number of batch in dataloader
    batches = len(dataloader)

    for batch in dataloader:
        for k, v in batch.items():
            batch[k] = v.to("cuda", non_blocking=True)
        outputs = model(**batch)

        for idx in range(num_layers):
            _features = _inputs[idx][-1]
            # abs
            # d2 = torch.mean(_features, dim=(0, 1))
            # sqrt of squared sum
            d2 = torch.sqrt(torch.sum(_features ** 2, dim=(0, 1)))
            importance[idx] += d2
            del _inputs[idx][-1]
    # neuron_importance / batches
    torch.div(importance, batches, out=importance)

    if structure == "H":
        for i in range(num_attention_heads):
            head_importance[:, i] = torch.sum(importance[:, i * attention_head_size: (i+1) * attention_head_size], dim=1) / attention_head_size
        importance = head_importance

    if type == "z-score":
        # standardize the importance
        means = torch.mean(importance, dim=1, keepdim=True)
        stds = torch.std(importance, dim=1, keepdim=True)
        standardized_importance = (importance - means) / stds
    elif type == "minmax":
        # normalize the importance
        maxs = torch.max(importance, dim=1, keepdim=True)[0]
        mins = torch.min(importance, dim=1, keepdim=True)[0]
        standardized_importance = (importance - mins) / (maxs - mins)
    elif type == "log":
        # log scaling the importance
        standardized_importance = torch.log(importance + 1e-8)
    elif type == "l2":
        # l2 norm normalization 
        standardized_importance = torch.div(importance, torch.norm(importance, p=2, dim=1, keepdim=True))
    else:
        raise ValueError("Invalid type of normalization")
    
    # remove hook function
    for handle in handles:
        handle.remove()
    del _inputs

    if log:
        print("d2-importance of structure {}: {} {}".format(structure, standardized_importance.shape, standardized_importance[:, :5]))

    return standardized_importance

def r2_neuron_importance(model, sigma, alpha, r, dataloader, log=False):
    """
    a function to compute r2 importance based on approximated convex hull of ffn2

    * Inputs
        - model: the target model to compress
        - sigma: width of the gaussian kernel
        - alpha: convergence rate of KCM
        - r: the rank of factorization
        - dataloader: dataloader for unlabeled data
    * Outputs
        - neuron_importance: importance of neurons, (layers x ffn dimension)
    """
    num_layers = model.config.num_hidden_layers
    ffn_dim = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    neuron_importance = torch.zeros(num_layers, ffn_dim).cuda()

    r = torch.tensor(r * hidden_size, dtype=torch.long)

    for l in range(num_layers):
        weight = get_ffn2(model, l).dense.weight
        # U, S, Vh = torch.linalg.svd(weight.T, full_matrices=False)
        # Ur = U[:, :r]
        # Sr = S[:r]
        # Vhr = Vh[:r, :]
        # new_weight = torch.matmul(Ur, Sr)
        c = torch.ones(ffn_dim, ffn_dim).cuda() * (1 / ffn_dim)
        # k = gaussian_kernel(new_weight, new_weight, sigma)
        k = gaussian_kernel(weight.T, weight.T, sigma)
        i = 0
        while True:
            # k(i, j): gaussian kernel of x_i and x_j
            # k(i): gaussian kernel of x_i with X
            next_c = torch.zeros(ffn_dim, ffn_dim).cuda()
            update = torch.sqrt(k / torch.matmul(k, c))
            next_c = c * update
            
            diff = torch.abs(torch.sum(next_c - c) / torch.sum(torch.abs(c)))
            # print("iter {} diff: {}, c: {}, next_c: {}, update: {}".format(i, diff, c[0, :5], next_c[0, :5], update[0, :5]))
            c = next_c
            if diff < alpha:
                break
            i += 1
        # r2_importance = diagnoal of c
        neuron_importance[l] = torch.diag(c)
    maxs = torch.max(neuron_importance, dim=1, keepdim=True)[0]
    mins = torch.min(neuron_importance, dim=1, keepdim=True)[0]
    standardized_importance = (neuron_importance - mins) / (maxs - mins) 
    if log:
        print("neuron r2-importance {} {}".format(standardized_importance.shape, standardized_importance[:, :5]))
    return standardized_importance

# gaussian kernel function
def gaussian_kernel(x, y, sigma):
    """
    a function to compute gaussian kernel between two matrix

    * Inputs
        - x: matrix (n x d)
        - y: matrix (m x d)
        - sigma: width of the gaussian kernel
    * Outputs
        - kernel: gaussian kernel matrix (n x m), each element is the similarity between x_i (i-th row of matrix x) and y_j (j-th row of matrix y)
    """

    # out = torch.zeros(x.size(0), x.size(0)).cuda()
    # for i in range(x.size(0)):
    #     for j in range(x.size(0)):
    #         # out[i, j] = torch.exp(-torch.sum((x[i] - y[j]) ** 2) / (2 * (sigma ** 2)))
    #         up = torch.exp(-torch.sum((x[i] - x[j]) ** 2) / (2 * (sigma ** 2)))
    #         down = torch.mul()
    # return out

    dist = torch.cdist(x, y, p=2)
    K = torch.exp(-dist.pow(2) / (sigma ** 2))
    # print("out: {} {}".format(K.shape, K[:5, :5]))
    return K
    
    # print("x: {} {}".format(x.shape, x[:10, :10]))
    # dist_matrix = torch.sum((x[:, None] - x) ** 2, dim=2)
    # Compute Gaussian kernel matrix
    # K = torch.exp(-dist_matrix / (2 * sigma**2))
    # print("out: {} {}".format(K.shape, K[:10, :10]))
    # return K


    # constant = 1 / (2 * np.pi * (sigma ** 2))
    # e = torch.exp(torch.div((torch.add(torch.pow(x, 2), torch.pow(y, 2))), 2 * (sigma ** 2)))
    # return torch.mul(constant, e)

def r2_head_importance(model, sigma, alpha, dataloader, log=False):
    """
    a function to compute r2 importance based on approximated convex hull of ffn2

    * Inputs
        - model: the target model to compress
        - sigma: width of the gaussian kernel
        - alpha: convergence rate of KCM
        - dataloader: dataloader for unlabeled data
    * Outputs
        - head_importance: importance of heads, (layers x attention head)
    """
    num_layers = model.config.num_hidden_layers
    num_attention_heads = model.config.num_attention_heads
    attention_head_size = int(model.config.hidden_size / num_attention_heads)
    hidden_size = model.config.hidden_size
    importance = torch.zeros(num_layers, hidden_size).cuda()
    head_importance = torch.zeros(num_layers, num_attention_heads).cuda()

    for l in range(num_layers):
        weight = get_mha_proj(model, l).dense.weight
        c = torch.ones(hidden_size, hidden_size).cuda() * (1 / hidden_size)
        k = gaussian_kernel(weight.T, weight.T, sigma)
        # print("layer {}, c = {}".format(l, c.shape))
        i = 0
        while True:
            # k(i, j): gaussian kernel of x_i and x_j
            # k(i): gaussian kernel of x_i with X
            next_c = torch.zeros(hidden_size, hidden_size).cuda()
            update = torch.sqrt(k / torch.matmul(k, c))
            next_c = c * update
            
            diff = torch.abs(torch.sum(next_c - c) / torch.sum(torch.abs(c)))
            # print("iter {} diff: {}, c: {}, next_c: {}, update: {}".format(i, diff, c[0, :5], next_c[0, :5], update[0, :5]))
            c = next_c
            if diff < alpha:
                break
            i += 1
        # r2_importance = diagnoal of c
        importance[l] = torch.diag(c)
        for i in range(num_attention_heads):
            head_importance[l, i] = torch.sum(importance[l, i * attention_head_size: (i+1) * attention_head_size]) / attention_head_size
        
       
    maxs = torch.max(head_importance, dim=1, keepdim=True)[0]
    mins = torch.min(head_importance, dim=1, keepdim=True)[0]
    standardized_importance = (head_importance - mins) / (maxs - mins)

    if log:
        print("head r2-importance {} {}".format(head_importance.shape, head_importance[:, :5])) 
    return standardized_importance

def collect_importance(model, args, sample_dataloader):
    head_d2_importance = d2_importance(model, args.d2_norm, "H", sample_dataloader, log=True)
    head_r2_importance = r2_head_importance(model, args.sigma, args.alpha, sample_dataloader, log=True)
    neuron_d2_importance = d2_importance(model, args.d2_norm, "N", sample_dataloader, log=True)
    neuron_r2_importance = r2_neuron_importance(model, args.sigma, args.alpha, args.r, sample_dataloader, log=True)
    neuron_importance = neuron_d2_importance * neuron_r2_importance
    head_importance = head_d2_importance * head_r2_importance
    return head_importance, neuron_importance

def rescale_mask_kcm(
    model,
    config,
    teacher_head_mask,
    teacher_neuron_mask,
    student_head_mask,
    student_neuron_mask,
    dataloader,
    classification_task=False,
):
    num_hidden_layers = config.num_hidden_layers
    rescaled_head_mask = student_head_mask.clone()
    rescaled_neuron_mask = student_neuron_mask.clone()

    for layer_idx in tqdm(range(num_hidden_layers)):
        teacher_inputs = collect_layer_inputs(
            model,
            teacher_head_mask,
            teacher_neuron_mask,
            layer_idx,
            prev_inputs=dataloader if layer_idx == 0 else teacher_inputs,
        )
        student_inputs = collect_layer_inputs(
            model,
            rescaled_head_mask,
            rescaled_neuron_mask,
            layer_idx,
            prev_inputs=dataloader if layer_idx == 0 else student_inputs,
        )

        if torch.count_nonzero(student_head_mask[layer_idx]) != 0 and layer_idx != 0:
            ATA, ATB = get_mha_lstsq(
                model,
                config,
                teacher_inputs,
                teacher_neuron_mask,
                student_inputs,
                rescaled_head_mask,
                rescaled_neuron_mask,
                layer_idx,
            )
            scale_factor, success = lsmr_cupy_solver(ATA, ATB)
            if not success:
                break
            scale_factor = scale_factor[:-1]
            if scale_factor.max() > 10 or scale_factor.min() < -10:
                break
            nonzero_heads = rescaled_head_mask[layer_idx].nonzero().flatten()
            for index, scale in zip(nonzero_heads, scale_factor):
                rescaled_head_mask[layer_idx][index] *= scale

        if torch.count_nonzero(student_neuron_mask[layer_idx]) != 0:
            cls_only = classification_task and (layer_idx == num_hidden_layers - 1)
            ATA, ATB = get_ffn_lstsq(
                model,
                config,
                teacher_inputs,
                teacher_neuron_mask,
                student_inputs,
                rescaled_head_mask,
                rescaled_neuron_mask,
                layer_idx,
                cls_only=cls_only,
            )
            scale_factor, success = lsmr_cupy_solver(ATA, ATB)
            if not success:
                break
            if scale_factor.max() > 10 or scale_factor.min() < -10:
                break
            nonzero_neurons = rescaled_neuron_mask[layer_idx].nonzero().flatten()
            for index, scale in zip(nonzero_neurons, scale_factor):
                rescaled_neuron_mask[layer_idx][index] *= scale

    return rescaled_head_mask, rescaled_neuron_mask