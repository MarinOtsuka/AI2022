# 以下を「model.py」に書き込み
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image

classes_ja = ["Tシャツ/トップス", "ズボン", "プルオーバー", "ドレス", "コート", "サンダル", "シャツ", "スニーカー", "バッグ", "ブーツ"]
classes_en = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
n_class = len(classes_ja)
img_size = 32

# CNNのモデル
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x

def predict(img):
    # モデルへの入力
    img = img.convert("RGB")
    img = img.resize((img_size, img_size))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))  
                                ])
    img = transform(img)
    x = img.reshape(1, 3, img_size, img_size)

    # 訓練済みモデル
    net = Net()
    net.load_state_dict(torch.load(
        "model_cnn.pth", map_location=torch.device("cpu")
        ))
    
    # 予測
    net.eval()
    y = net(x)

    # 結果を返す
    y_prob = torch.nn.functional.softmax(torch.squeeze(y))  # 確率で表す
    sorted_prob, sorted_indices = torch.sort(y_prob, descending=True)  # 降順にソート
    return [(classes_ja[idx], classes_en[idx], prob.item()) for idx, prob in zip(sorted_indices, sorted_prob)]
