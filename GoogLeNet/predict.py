import torch
from model import GoogLeNet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
# 数据预处理
data_transform = transforms.Compose(
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load image
img = Image.open("../ViT/images/banded.jpg").convert("RGB")
plt.imshow(img)
# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)

# read class_indict
try:
    json_file = open('imagenet.json', 'r')
    # 将json转化为字典
    class_indict = json.load(json_file)
except Exception as e:
    print(e)
    exit(-1)

# create model
model = GoogLeNet(num_classes=1000, aux_logits=False)
# load model weights
model_weight_path = "weights/googlenet.pth"
missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)
model.eval()
with torch.no_grad():
    # predict class
    # output = torch.squeeze(model(img))
    output = model(img)
    print(output.shape)
    predict = torch.softmax(output.squeeze(), dim=0)
    predict_cla = torch.argmax(predict).numpy()
# # 打印预测种类和概率值
print(class_indict[int(predict_cla)], predict[predict_cla].item())
plt.show()
#




