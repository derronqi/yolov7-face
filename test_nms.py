import torch
import torchvision

def test_nms(N=128):
    boxes = []
    scores = []
    
    for n in range(N):
        boxes.append((n, n+1, n, n+1))
        scores.append(n)

    boxes = torch.Tensor(boxes)
    scores = torch.Tensor(scores)
    indices = torchvision.ops.nms(boxes, scores, 0.5)

def test_nms_cuda(N=128):
    boxes = []
    scores = []
    
    for n in range(N):
        boxes.append((n, n+1, n, n+1))
        scores.append(n)

    boxes = torch.Tensor(boxes).cuda()
    scores = torch.Tensor(scores).cuda()
    indices = torchvision.ops.nms(boxes, scores, 0.5)
    
test_nms()
print("ops nms cpu success")
test_nms_cuda()
print("ops cuda success")
