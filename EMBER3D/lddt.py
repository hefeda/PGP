import torch


def lddt(dist_pred,  # (L x L)
         dist_gt):  # (L x L)

    length = dist_pred.shape[-1]

    cutoff = 15.0
    mask_2d = (dist_gt < cutoff) * (1.0 - torch.eye(length, device=dist_gt.device))

    dist_l1 = torch.abs(dist_pred - dist_gt)

    lddt_score = (dist_l1 < 0.5).type(dist_l1.dtype) + \
                 (dist_l1 < 1.0).type(dist_l1.dtype) + \
                 (dist_l1 < 2.0).type(dist_l1.dtype) + \
                 (dist_l1 < 4.0).type(dist_l1.dtype)
    lddt_score *= 0.25
    lddt_score *= mask_2d

    lddt_score = (1e-10 + torch.sum(lddt_score, dim=(-1,))) / \
                 (1e-10 + torch.sum(mask_2d, dim=(-1,)))
    return lddt_score
