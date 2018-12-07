# CIFAR-100--Knowledge-Distillation-with-Augmented-Data
Knowledge distillation using CIFAR 100

Paper : https://arxiv.org/abs/1503.02531

# Teacher Model
Resnet 20 (you can change with 50 or others)

# Student Model
MobileNet or CNN - LSTM
<div class="imgTopic">
 <h1 class="title"><a href="#">CNN - LSTM model</a></h1>
 <p class="content"><a href="#"><img src="https://user-images.githubusercontent.com/29685163/49659290-3f0ac700-fa87-11e8-815b-d0fb6d7b7d25.png" alt="" width = "675" height ="350"/></a></p>
</div>

# Data
ImageNet Dataset : http://www.image-net.org/

# How to get dataset?
Link = http://hpkim0512.blogspot.com/2017/12/sdf.html?m=1

# Training Method
Stop Training if overfitting(validation set accuracy doesn't increase) happens more than 50 epochs

# Programming Language
Python 3.6
Tensorflow, keras

# OS dependency
windows 10, ubuntu linux

# Result

<div class="imgTopic2">
 <h1 class="title"><a href="#">Resnet - 20's Training result</a></h1>
 <p class="content"><a href="#"><img src="https://user-images.githubusercontent.com/29685163/49659285-3b774000-fa87-11e8-891a-76f6312eddd0.png" alt="" width = "675" height ="350"/></a></p>
</div>

<div class="imgTopic3">
 <h1 class="title"><a href="#">Training result with spatial complexity</a></h1>
 <p class="content"><a href="#"><img src="https://user-images.githubusercontent.com/29685163/49659292-40d48a80-fa87-11e8-9252-bedda096fb40.png" alt="" width = "675" height ="350"/></a></p>
</div>

<div class="imgTopic4">
 <h1 class="title"><a href="#">Training result with time complexity</a></h1>
 <p class="content"><a href="#"><img src="https://user-images.githubusercontent.com/29685163/49659295-4205b780-fa87-11e8-93fc-bbe1d04ab2a1.png" alt="" width = "675" height ="350"/></a></p>
</div>
