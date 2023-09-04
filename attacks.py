import torch
import torch.nn.functional as F
from utils import label_to_onehot
from torchvision import transforms

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

criterion = torch.nn.CrossEntropyLoss()
tp = transforms.ToTensor()
tt = transforms.ToPILImage()
history_DLG = []


def DLG(model, dst, idx, num_iters, template_idx):
    print("============Deep Leakage from Gradients============")
    # Get the ground-truth image and its label
    gt_image, gt_label = dst.__getitem__(idx)
    gt_data = gt_image.to(device)
    gt_data = gt_data.view(1, 128, 128)
    gt_label = gt_label.long().to(device)
    gt_onehot_label = label_to_onehot(gt_label, num_classes=2)
    gt_onehot_label = gt_onehot_label.float()
    # Compute the original gradients
    model = model.float()
    model.eval()
    pred = model(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, model.parameters())
    original_dy_dx = list((_.detach().clone() for _ in dy_dx))  # Deep copy

    '''Generate the dummy data and dummy label'''
    # template_data, template_label = dst.__getitem__(template_idx)
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    # dummy_data = template_data
    dummy_data = dummy_data.view(1, *dummy_data.size())
    dummy_data = dummy_data.float()
    dummy_data = dummy_data.to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)
    # template_label = template_label.long()
    # dummy_onehot_label = label_to_onehot(template_label, num_classes=2)  # (1, 64)
    # dummy_label = dummy_onehot_label.to(device).requires_grad_(True)
    print("The initial dummy image: ")
    print(dummy_data)
    # The L-BFGS method is a type of second-order optimization algorithm and belongs to a class of Quasi-Newton methods.
    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    for iter in range(num_iters):
        def closure():
            optimizer.zero_grad()
            dummy_pred = model(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, model.parameters(), create_graph=True)
            # Calculate the difference between the dummy RSRP matrix and the ground-truth RSRP matrix using the Euclidean distance
            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()
            return grad_diff

        optimizer.step(closure)
        if iter % 50 == 0:
            current_loss = closure()
            print(iter, "%.4f" % current_loss.item())
            print(dummy_data)
            history_DLG.append(dummy_data)

    return dummy_data, dummy_label
