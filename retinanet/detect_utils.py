import torch 
from torchvision.ops import nms
from retinanet.utils import BBoxTransform, ClipBoxes


def detect_from_pred_boxes(input_img, classifications, regressions, anchors, class_embeddings, n_seen, detect_type, 
                           scorethresh = 0.05, nmsthresh = 0.5, nboxes = 100):
    
    if detect_type == 'zsd':
        u_classifications = unseen_inference(classifications, class_embeddings, n_seen)
        classifications = torch.zeros(classifications[:,:,:n_seen].shape).cuda()
        classifications = torch.cat([classifications, u_classifications], dim = -1)
    elif detect_type == 'gzsd':
        u_classifications = unseen_inference(classifications, class_embeddings, n_seen)
        classifications = classifications[:,:,:n_seen]
        classifications = torch.cat([classifications, u_classifications], dim = -1)
    else:
        classifications = classifications[:,:,:n_seen]            
    # breakpoint()
    #####################################    
    ############# INFERENCE #############
    #####################################  
    regressBoxes = BBoxTransform()
    clipBoxes = ClipBoxes()
    transformed_anchors = regressBoxes(anchors, regressions)
    transformed_anchors = clipBoxes(transformed_anchors, input_img)

    finalScores = torch.Tensor([])
    finalAnchorBoxesIndexes = torch.Tensor([]).long()
    finalAnchorBoxesCoordinates = torch.Tensor([])

    if torch.cuda.is_available():
        finalScores = finalScores.cuda()
        finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
        finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

    for i in range(classifications.shape[2]):
        scores = torch.squeeze(classifications[:, :, i])
        scores_over_thresh = (scores > scorethresh)
        if scores_over_thresh.sum() == 0:
            # no boxes to NMS, just continue
            continue

        scores = scores[scores_over_thresh]
        anchorBoxes = torch.squeeze(transformed_anchors)
        anchorBoxes = anchorBoxes[scores_over_thresh]
        
        if nboxes is not None:
            try:
                scores, idx = torch.topk(scores, nboxes, dim = 0)
                anchorBoxes = anchorBoxes[idx]
            except:
                pass 
        # breakpoint()
        labels = torch.tensor([i] * scores.shape[0])
        if torch.cuda.is_available():
            labels = labels.cuda()

        anchors_nms_idx = nms(anchorBoxes, scores, nmsthresh)
        scores = scores[anchors_nms_idx]
        anchorBoxes = anchorBoxes[anchors_nms_idx]
        labels = labels[anchors_nms_idx]

        finalScores = torch.cat((finalScores, scores))
        finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, labels))
        finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes))

    # breakpoint()
    if finalScores.shape[0] > 0:
        final_nms_idx = nms(finalAnchorBoxesCoordinates, finalScores, nmsthresh)
        finalScores = finalScores[final_nms_idx]
        finalAnchorBoxesIndexes = finalAnchorBoxesIndexes[final_nms_idx]
        finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates[final_nms_idx]
    return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]

def unseen_inference(classifications, class_embeddings, n_seen):
    #### Inferring unseen scores ####
    class_embeddings_tensor = torch.from_numpy(class_embeddings).float()
    if torch.cuda.is_available():
        class_embeddings_tensor = class_embeddings_tensor.cuda()
    emb_size = class_embeddings_tensor.shape[1]
    batch_size = classifications.shape[0]
    u_classifications = []
    word_seen = class_embeddings_tensor[:n_seen,:]
    word_unseen = class_embeddings_tensor[n_seen:,:]
    for j in range(batch_size):

        # u_cls = classifications[j, :, :]
        # u_cls = u_cls[:,:n_seen]
        # T = 5

        # mask = torch.ones_like(u_cls,dtype=torch.float32).cuda()
        # mask[:, T:] = 0.0
        # sorted_u_cls, sorted_u_cls_arg = torch.sort(-u_cls, dim=1)
        # sorted_u_cls = -sorted_u_cls
        # sorted_u_cls = sorted_u_cls * mask
        # restroed_score = mask
        # for i in range(u_cls.shape[0]):
        #     restroed_score[i, sorted_u_cls_arg[i, :]] = sorted_u_cls[i, :]

        # unseen_pd = restroed_score @ word_seen
        # unseen_scores = unseen_pd @ word_unseen.T
        # u_classifications.append(unseen_scores) 

        u_cls = classifications[j, :, :]
        u_cls = u_cls[:,:n_seen]
        
        topT_scores, topT_idx = torch.topk(u_cls, k=5, dim =-1) # p'
        W_topT = class_embeddings_tensor.repeat(u_cls.shape[0], 1, 1) # W'
        topT_idx = topT_idx.unsqueeze(-1).repeat(1,1, emb_size)
        W_topT = torch.gather(W_topT, 1, topT_idx)

        W_u = class_embeddings_tensor[n_seen:,:].repeat(u_cls.shape[0],1,1) # Wu
        u_scores = topT_scores.unsqueeze(1) @ W_topT @ W_u.permute(0,2,1)
        u_scores = u_scores.squeeze(1)
        # u_scores = torch.nn.functional.tanh(u_scores)
        u_classifications.append(u_scores) 

        # breakpoint()
    # u_classifications = torch.zeros(u_scores.unsqueeze(0).shape).cuda()
    u_classifications = torch.stack(u_classifications)

    return u_classifications