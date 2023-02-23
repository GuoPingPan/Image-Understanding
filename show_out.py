import tkinter as tk
import ttkbootstrap as ttk
import tkinter.filedialog
from PIL import Image,ImageTk
import matplotlib.pyplot as plt
from ttkbootstrap.constants import *
from typing import Any

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os,json,time
from GoogLeNet.model import GoogLeNet
from ResNet.model import resnet34,resnet50
from VGG.model import vgg
from ViT.model import vit_base_patch16_224




class ShowOut:
    def __init__(self):
        self.window = ttk.Window(title='Network Software',
                             themename='litera',
                             size=(770,600),
                             resizable=None)
        self.img_frame = ttk.Frame(self.window,width=480,height=360,bootstyle='info')
        self.img_frame.place(x=10,y=10)
        self.net_frame = ttk.Frame(self.window,width=240,height=360,bootstyle='info')
        self.net_frame.place(x=520,y=10)
        self.output_frame = ttk.Frame(self.window,width=750,height=180,bootstyle='info')
        self.output_frame.place(x=10,y=390)
        self.output = ttk.StringVar()

        self.data_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.input = None
        self.create_models()
        self.model = self.model_vit16_b

        json_path = 'ResNet/imagenet.json'
        assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
        with open(json_path, "r") as f:
            self.class_indict = json.load(f)

        self.run()

    def create_models(self):
        model_weight_pth = [
            'VGG/weights/vgg16pre.pth',
            'VGG/weights/vgg19-dcbb9e9d.pth',
            'ViT/weights/jx_vit_base_p16_224-80ecf9dd.pth',
            'GoogLeNet/weights/googlenet.pth',
            'ResNet/weights/resnet34.pth',
            'ResNet/weights/resnet50.pth',
        ]
        self.model_vgg16 = vgg(model_name='vgg16').to(self.device)
        self.model_vgg19 = vgg(model_name='vgg19').to(self.device)
        self.model_vit16_b = vit_base_patch16_224(num_classes=1000).to(self.device)
        self.model_googlenet = GoogLeNet(num_classes=1000,aux_logits=False).to(self.device)
        self.model_resnet34 = resnet34(num_classes=1000).to(self.device)
        self.model_resnet50 = resnet50(num_classes=1000).to(self.device)

        self.model_list = [
            self.model_vgg16,
            self.model_vgg19,
            self.model_vit16_b,
            self.model_googlenet,
            self.model_resnet34,
            self.model_resnet50
        ]

        for pth,model in zip(model_weight_pth,self.model_list):
            # print(pth,model.__str__())
            assert os.path.exists(pth), f"weight file [{pth}] doesn't exists."
            model.load_state_dict(torch.load(pth,map_location=self.device),strict=False)


    def create_img_frame(self):
        # canvas = tk.Canvas(window,width=480,height=360)
        l1 = ttk.Label(master=self.img_frame,bootstyle='light')
        empty = Image.new(mode='RGB',size=(480,360))
        l1.config(image=ImageTk.PhotoImage(empty))
        l1.place(x=0,y=0)

        def choose_photo():
            path = tk.filedialog.askopenfilename()
            img = Image.open(path)

            ''' input '''
            self.input = self.data_transform(img).unsqueeze(0)

            ''' show '''
            img = img.resize((480, 360))
            img = ImageTk.PhotoImage(img)
            l1.config(image=img)
            l1.image = img
            # img = canvas.create_image(0,0,anchor='nw',image=img)
            # plt.show()

        # canvas.place(x=10,y=10)
        # l1.pack()
        b1 = ttk.Button(self.img_frame, width=10, text='选择图片', command=choose_photo,bootstyle='primary')
        b1.place(x=360, y=320)


    def create_net_frame(self):
        var2 = ttk.StringVar()
        var2.set(0)
        sub1 = ttk.Frame(self.net_frame,width=240,height=100,bootstyle='light')
        sub1.pack(side=tk.TOP)
        l2 = ttk.Label(sub1,width=30,text='Empty',bootstyle='info')
        l2.pack()

        net_list = [
            'Vgg16',
            'Vgg19',
            'ViT',
            'GoogLeNet',
            'ResNet34',
            'ResNet50',
        ]
        def selection():
            sig = tk.messagebox.askyesno('',message='确定要更换网络['+var2.get()+']?')
            if sig:
                l2.config(text='You had selected [' + var2.get()+']')

                net = var2.get()
                print(net)
                for i,model in enumerate(net_list):
                    if net == model:
                        self.model = self.model_list[i]

        for i,net in enumerate(net_list):
            b = ttk.Radiobutton(self.net_frame,width=30,text=net,variable=var2,bootstyle='success',value=net,command=selection)
            b.pack(ipady=5)


    def create_output_frame(self):

        def update_output():
            # self.output.set(output)
            show = ''
            if self.input is None:
                print(f"you haven't load the image yet.")
                return
            torch.cuda.empty_cache()
            self.model.eval()
            with torch.no_grad():
                output = self.model(self.input.to(self.device)).squeeze()
                output = torch.softmax(output,dim=-1)
                top5_values,top5_indices = torch.topk(output,5,dim=-1)
                # print(sorted(output.cpu().numpy().tolist(),reverse=True))
                for i in range(top5_indices.size(0)):
                    # result[self.class_indict[top5_indices[i]].replace('\n','')] = \
                    #     top5_values[i]
                    class_ = self.class_indict[top5_indices[i]].replace('\n','')
                    show += f"{class_}      prob:{top5_values[i]:<5f}\n"

            t3.delete(1.0, 'end')
            t3.insert('insert', show)




        b3 = ttk.Button(self.output_frame,text='Inference',width=10,bootstyle='primary',command=update_output)
        b3.place(x=620,y=10)
        # t3 = tk.Text(self.output_frame,borderwidth=4)
        # t3.place(x=20,y=20)
        t3 = tk.Text(self.output_frame,width=50,height=6,borderwidth=2)
        t3.insert('insert','Output')
        t3.place(x=10,y=18)

        fg3 = ttk.Floodgauge(
            self.output_frame,
            bootstyle=SUCCESS,
            length=100,
            maximum=101,
            orient=VERTICAL,
            font=("微软雅黑", 10, 'bold'),
            mask='loading\n{}%',
        )
        fg3.place(x=650,y=60)
        fg3.start()
        # fg3.configure(value=20)
        # fg3.step(50)
        # fg3.stop()

        fg3.step(99)
        time.sleep(0.1)
        fg3.stop()


    def run(self):
        self.create_img_frame()
        self.create_net_frame()
        self.create_output_frame()
        self.window.mainloop()





show = ShowOut()
