import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

import model as vit




def main():
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img_path = 'images/banded.jpg'
    assert os.path.exists(img_path), f"image file [{img_path}] doesn't exist."
    img = Image.open(img_path)
    plt.imshow(img)
    img = data_transform(img).unsqueeze(0)

    model = vit.vit_base_patch16_224().to(device)
    weight_path = 'weights/jx_vit_base_p16_224-80ecf9dd.pth'
    ckpt = torch.load(weight_path,map_location=device)
    model.load_state_dict(ckpt,strict=False)

    def get_parameter_number(net):
        total_num = sum(p.numel() for p in net.parameters())
        trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
        return {'Total': total_num, 'Trainable': trainable_num}
    #
    # print(get_parameter_number(model))
    #
    with torch.no_grad():
        output = model(img.to(device))
        print(output.shape)
        pred = torch.softmax(output,dim=-1).argmax()

    print(f"class:{pred}")



if __name__ == '__main__':
    main()