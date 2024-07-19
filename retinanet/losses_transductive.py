import numpy as np
import torch
import torch.nn as nn

def calc_iou(a, b):
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = torch.min(torch.unsqueeze(a[:, 2], dim=1), b[:, 2]) - torch.max(torch.unsqueeze(a[:, 0], 1), b[:, 0])
    ih = torch.min(torch.unsqueeze(a[:, 3], dim=1), b[:, 3]) - torch.max(torch.unsqueeze(a[:, 1], 1), b[:, 1])

    iw = torch.clamp(iw, min=0)
    ih = torch.clamp(ih, min=0)

    ua = torch.unsqueeze((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), dim=1) + area - iw * ih

    ua = torch.clamp(ua, min=1e-8)

    intersection = iw * ih

    IoU = intersection / ua

    return IoU

class TransductiveLoss(nn.Module):
    def __init__(self, n_seen, alpha = 0.25, gamma = 2.0, eta = 1.0, beta = 0.3, th = 0.3):
        super().__init__()
        self.n_seen = n_seen
        self.alpha = alpha
        self.gamma = gamma
        self.eta = eta 
        self.beta = beta 
        self.th = th 
    def forward(self, classifications, anchors, annotations):

        batch_size = classifications.shape[0]
        fixed_losses = []
        dynamic_seen_losses = []
        dynamic_unseen_losses = []
        for j in range(batch_size):

            classification = classifications[j, :, :]
            classification = torch.clamp(classification, 1e-5, 1.0 - 1e-5)

            ###################################
            ### compute the fixed component ###
            ###################################
            
            seen_classification = classification[:,:self.n_seen] # get scores for seen classes only
            bbox_annotation = annotations[j, :, :]
            #breakpoint()
            if bbox_annotation.shape[0] == 0: # avoid no bbox situation
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(seen_classification.shape).cuda() *self.alpha
                else:
                    alpha_factor = torch.ones(seen_classification.shape) *self.alpha

                alpha_factor = 1. - alpha_factor
                focal_weight = seen_classification
                focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)

                bce = -(torch.log(1.0 - seen_classification))

                cls_loss = focal_weight * bce
                fixed_losses.append(cls_loss.sum())
                #breakpoint()
            else:

                bbox_annotation = bbox_annotation[bbox_annotation[:, 4] != -1] 

                IoU = calc_iou(anchors[0, :, :], bbox_annotation[:, :4]) # num_anchors x num_annotations
                #breakpoint()
                IoU_max, IoU_argmax = torch.max(IoU, dim=1) # num_anchors x 1

                # create a target mask for the fixed loss
                targets = torch.ones(seen_classification.shape) * -1

                if torch.cuda.is_available():
                    targets = targets.cuda()

                targets[torch.lt(IoU_max, 0.4), :] = 0

                positive_indices = torch.ge(IoU_max, 0.5)

                num_positive_anchors = positive_indices.sum()

                assigned_annotations = bbox_annotation[IoU_argmax, :]

                targets[positive_indices, :] = 0
                targets[positive_indices, assigned_annotations[positive_indices, 4].long()] = 1

                # compute fixed loss
                if torch.cuda.is_available():
                    alpha_factor = torch.ones(targets.shape).cuda() *self.alpha
                else:
                    alpha_factor = torch.ones(targets.shape) *self.alpha
                alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
                focal_weight = torch.where(torch.eq(targets, 1.), 1. - seen_classification, seen_classification)
                focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
                bce = -(targets * torch.log(seen_classification+1e-10) \
                        + (1.0 - targets) * torch.log(1.0 - seen_classification+1e-10))

                cls_loss = focal_weight * bce
                if torch.cuda.is_available():
                    cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
                else:
                    cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

                fixed_losses.append(cls_loss.sum()/torch.clamp(num_positive_anchors.float(), min=1.0))

            #####################################
            ### compute the seen dynamic loss ###
            #####################################

            # create a target mask for the seen dynamic loss
            targets = torch.where(classification[:,:self.n_seen] > self.th, 1., 0.)
            if torch.cuda.is_available():
                targets = targets.cuda()
            num_positive_anchors_dynamic_seen = (torch.max(targets, dim =-1)[0] > 0).sum()
            
            # compute dynamic seen loss
            if torch.cuda.is_available():
                alpha_factor = torch.ones(targets.shape).cuda() *self.alpha
            else:
                alpha_factor = torch.ones(targets.shape) *self.alpha

            alpha_factor = torch.where(torch.eq(targets, 1.), alpha_factor, 1. - alpha_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification[:,:self.n_seen], classification[:,:self.n_seen])
            focal_weight = alpha_factor * torch.pow(focal_weight, self.gamma)
            bce = -(targets * torch.log(classification[:,:self.n_seen]+1e-10) \
                    + (1.0 - targets) * torch.log(1.0 - classification[:,:self.n_seen]+1e-10))

            cls_loss = focal_weight * bce
            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))
            
            cls_loss /= torch.clamp(num_positive_anchors_dynamic_seen.float(), min=1.0)

            dynamic_seen_losses.append(cls_loss.sum())

            #######################################
            ### compute the unseen dynamic loss ###
            #######################################

            # create a target mask for the seen dynamic loss
            targets = torch.where(classification[:,self.n_seen:] > self.th, 1., 0.)
            if torch.cuda.is_available():
                targets = targets.cuda()
            num_positive_anchors_dynamic_unseen = (torch.max(targets, dim =-1)[0] > 0).sum()

            # compute dynamic unseen loss
            if torch.cuda.is_available():
                beta_factor = torch.ones(targets.shape).cuda() * self.beta
            else:
                beta_factor = torch.ones(targets.shape) * self.beta

            beta_factor = torch.where(torch.eq(targets, 1.), beta_factor, 1. - beta_factor)
            focal_weight = torch.where(torch.eq(targets, 1.), 1. - classification[:,self.n_seen:], classification[:,self.n_seen:])
            focal_weight = beta_factor * torch.pow(focal_weight, self.eta)

            bce = -(targets * torch.log(classification[:,self.n_seen:]*classification[:,self.n_seen:]+1e-10) \
                    + (1.0 - targets) * torch.log((1.0 - classification[:,self.n_seen:])*classification[:,self.n_seen:]+1e-10))

            cls_loss = focal_weight * bce

            if torch.cuda.is_available():
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape).cuda())
            else:
                cls_loss = torch.where(torch.ne(targets, -1.0), cls_loss, torch.zeros(cls_loss.shape))

            cls_loss /= torch.clamp(num_positive_anchors_dynamic_unseen.float(), min=1.0)

            dynamic_unseen_losses.append(cls_loss.sum())

        fixed_losses = torch.stack(fixed_losses).mean(dim=0, keepdim=True)
        dynamic_seen_losses = torch.stack(dynamic_seen_losses).mean(dim=0, keepdim=True)
        dynamic_unseen_losses = torch.stack(dynamic_unseen_losses).mean(dim=0, keepdim=True)

        return fixed_losses, dynamic_seen_losses, dynamic_unseen_losses

