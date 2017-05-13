# chinese_recongition
Handwritten recognition of Chinese characters

## DateSet
[BaiduYun](https://pan.baidu.com/s/1o84jIrg#list/path=%2F)
+ train images number:1787K+
+ test images number:10K+

---

## NetWork
five layers: 3 conv_layers + 2 full_connection_layers

---

## Train
you should run `training.py`
**PS**:
edit `training.py` like below:
``` python
if __name__=='__main__':
    run_training()
    #test()
    # stri = '/home/saverio_sun/project/chinese_rec_data/train/00547/40187.png'
    # evaluate_one_pic(stri)
```

---

## Test
you should run `training.py`
it has two ways to test accuracy
1. **test lots of images**
edit `training.py` like below:
``` python
if __name__=='__main__':
    #run_training()
    test()
    # stri = '/home/saverio_sun/project/chinese_rec_data/train/00547/40187.png'
    # evaluate_one_pic(stri)
```
2. **test one image**
edit `training.py` like below:
``` python
if __name__=='__main__':
    #run_training()
    #test()
    stri = '/home/saverio_sun/project/chinese_rec_data/train/00547/40187.png'
    evaluate_one_pic(stri)
```
    
## reference：
[小石头的码疯窝](http://hacker.duanshishi.com/?p=1753)
