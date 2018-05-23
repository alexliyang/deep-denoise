# deep-denoise
[tensorflow](https://www.tensorflow.org)1.4下跑的代码，代码修改自zsdonghao的SRGAN，推荐去看下他的[TensorLayer](http://tensorlayer.readthedocs.io/en/latest/)
这里作为入门DL的经验总结和笔记吧

GAN真的是个天才的想法

我在里面加了个GroupNorm，因为跑大图片跑不动大batch，小batch train出来的效果炸了，LN和IN效果也都一般，刚好看到kaiminghe大神的GN，加进去试了试，收敛的确实比前面两者要好很多，伪影少了，色彩更舒服了。

因为要过一张7500×3000的球目照片，我把网络参数削减了，不知道之后训练集大了会不会欠拟合。

前两层stride取2作为下采样， 后面可以用subpixleconv 或者upsizeconv作为上采样，我感觉差不多两者效果，但upsizeconv训练起来巨慢，别用deconv。

伪影问题还是很严重，数据集也好难收集，“garbage in， garbage out”，认真整理训练集。



