# 计算机视觉实践-练习05-视差匹配

## 1. 视差匹配






## 2. 实验说明


首先读入图像，使用SIFT检测器来获取图像的关键点

```python
# 使用SIFT
sift = cv2.SIFT_create()
keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
```
然后使用FLANN匹配器进行匹配

```python
# FLANN匹配器
index_params = dict(algorithm=0, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
rawMatches = flann.knnMatch(descriptors1, descriptors2, k=2)
```

最后通过匹配点对，计算单应性矩阵

```python
# 获取匹配点的坐标
pts1 = np.float32([keypoints1[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
pts2 = np.float32([keypoints2[m.trainIdx].pt for m in matches]).reshape(-1,1,2)

# 计算Homography矩阵
H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
```

可以将单应性矩阵应用于图像来观察效果。





## 3. 结果

打印单应性矩阵，详细内容位于jupyter notebook文件中。使用单应性矩阵对图像进行变换，得到的结果保存于``./image/res.jpg``

[.ipynb文件](./assignment03_SR.ipynb)


![res](./md_img/comp.png)


结果显示，视角变换符合给出的图像，可以继续执行图像拼接或融合等操作



## 运行说明

于jupyter notebook运行全部cell即可。单应性矩阵和对比结果会打印出来，变换后的图像会存储于``./image/res.jpg``





