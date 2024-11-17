import json
import matplotlib.pyplot as plt

# 저장된 손실 값 불러오기
with open("train_loss_log.json", "r") as f:
    train_losses = json.load(f)

with open("val_loss_log.json", "r") as f:
    val_losses = json.load(f)

# 에포크 단위로 손실 값 시각화
epochs = range(1, len(train_losses) + 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
