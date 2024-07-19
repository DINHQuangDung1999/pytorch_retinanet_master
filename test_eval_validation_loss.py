import os 
os.chdir('./pytorch_retinanet_master')
import torch
from tqdm import tqdm 
from retinanet.losses import FocalLoss
from torchvision import transforms
from retinanet.dataloader import CocoDataset, collater, Resizer, Normalizer
from torch.utils.data import DataLoader
import wandb

def main():
    dataset = CocoDataset('../data/PascalVOC', set_name='testseen2007zsd', is_zsd=True, # load only seen classes
                            transform=transforms.Compose([Normalizer(), Resizer()]))
    dataloader = DataLoader(dataset, num_workers=3, collate_fn=collater)

    loss_fn = FocalLoss()

    # breakpoint()
    wandb.login()
    wandb.init(
        # set the wandb project where this run will be logged
        project = "Traditional",
        id      = "traditional_0",
        resume  = "allow",
        # track hyperparameters and run metadata
        config={
            "batch_size": 4,
            "learning_rate": 1e-5,
            "dataset": 'pascalvoc',
            "epochs": 30,
            }
        )

    with torch.no_grad():
        epoch_loss = []
        for i in range(30):
            model_path = f'/home/qdinh/pytorch_retinanet_master/checkpoints/pascalvoc/traditional/traditional_resnet50_{i}.pt'
            #### Create the model ####
            if torch.cuda.is_available():
                detector = torch.load(model_path)
                detector = torch.nn.DataParallel(detector).cuda()
            else:
                detector = torch.load(model_path, map_location=torch.device('cpu'))

            for index, data in enumerate(tqdm(dataloader)):
                inputs = data['img']
                # run network
                if torch.cuda.is_available():
                    classifications, regressions, anchors = detector(inputs.cuda())
                    annotations = data['annot'].cuda()
                else:
                    classifications, regressions, anchors = detector(inputs)
                    annotations = data['annot']
                # breakpoint()
                classification_loss, regression_loss = loss_fn(classifications, regressions, anchors, annotations)
                
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                
                loss = classification_loss + regression_loss
                epoch_loss.append(loss)
                wandb.log({'val_cls_loss' : float(classification_loss),
                            'val_reg_loss'  : float(regression_loss),
                            'val_total_loss': float(loss)})
            wandb.log({'val_epoch_total_loss': float(torch.mean(epoch_loss))})
if __name__ == '__main__':
    main()