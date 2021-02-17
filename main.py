import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import torch.optim as optim
from torchvision.utils import save_image
from tqdm import tqdm

class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.chosen_features=['0','5','10','19','28']
        self.model=models.vgg19(pretrained=True).features[:29]
    def forward(self,x):
        features=[]
        for layer_num,layer in enumerate(self.model):
            x=layer(x)
            if str(layer_num) in self.chosen_features:
                features.append(x)
        
        return features

def load_image(image_name):
    image=Image.open(image_name)
    image=loader(image).unsqueeze(dim=0)
    return image.to(device)


#image prepare
print("initializing")
device=torch.device("cuda" if torch.cuda.is_available else "cpu")
image_size=356
loader=transforms.Compose(
    [
        transforms.Resize((image_size,image_size)),
        transforms.ToTensor()
    ]
)
content_img=load_image("content\Image3.jpg")
style_img=load_image("style\sky.jpg")
generated_img=content_img.clone().requires_grad_(True)
model=VGG().to(device)
#hyperparameters
total_steps=6000
learning_rate=0.001
alpha=1 #content loss
beta=0.1 #style loss
optimizer=optim.Adam([generated_img],lr=learning_rate)

counter=1
for step in tqdm(range(total_steps)):
    generated_features=model(generated_img)
    content_img_features=model(content_img)
    style_img_features=model(style_img)

    style_loss=content_loss=0

    for gen_features,cont_features,style_features in zip(
        generated_features,content_img_features,style_img_features
    ):
        batch_size,channel,height,width=gen_features.shape
        #content loss
        content_loss+=torch.mean((gen_features-cont_features)**2)
        #compute Style Matrix
        G= gen_features.view(channel,height*width).mm(
            gen_features.view(channel,height*width).t()
        )

        A= style_features.view(channel,height*width).mm(
            style_features.view(channel,height*width).t()
        )
        #style loss
        style_loss+=torch.mean((G-A)**2)
    total_loss=alpha*content_loss+beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    if step % 200==0:
        print(total_loss)
        save_image(generated_img,"generated\image"+str(counter)+".png")
        counter+=1