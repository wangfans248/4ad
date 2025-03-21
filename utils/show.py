from PIL import Image
import matplotlib.pyplot as plt
import numpy as np 


# 读取图片
img = Image.open('data/MVTecAD/capsule/test/crack/000.png')
gray_img = img.convert('L')  # 转为灰度图
gray_img = np.array(gray_img)  # 转为 NumPy 数组
# 显示图像
print(gray_img[600:620, 840:880])
plt.imshow(gray_img, cmap='gray')
plt.title("Image with Defects")
plt.show()
