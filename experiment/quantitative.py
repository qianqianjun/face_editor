"""
定量分析实验
检测人脸相似度的实验
"""
import requests
import base64
import json
import time
import numpy as np
import matplotlib.pyplot as plt
class image(object):
    def __init__(self, coders:str, image_type:str="BASE64", face_type:str="LIVE"):
        """
        设置百度AI开放平台的数据输入标准
        :param coders:  图片的base64编码
        :param image_type:  图像类型
        :param face_type:  人脸类型
        """
        self.image=coders
        self.image_type=image_type
        self.face_type=face_type
        self.liveness_control="NONE" # 不进行活体检测
    def toJSON(self):
        temp={
            "image":self.image,
            "image_type":self.image_type,
            "face_type":self.face_type,
            "liveness_control":self.liveness_control
        }
        return temp # 将图片转换为百度AI接口的标准输入
class Data(object):
    def __init__(self,path1,path2):
        """
        将原图和编辑结果进行相似度比较
        :param path1: 原图所在目录
        :param path2: 编辑结果所在目录
        """
        self.images=[]
        with open(path1,"rb") as f:
            encoder=base64.b64encode(f.read())
            text=encoder.decode()
            self.images.append(image(text))
        with open(path2,"rb") as f:
            encoder=base64.b64encode(f.read())
            text=encoder.decode()
            self.images.append(image(text))
    def toJSON(self):
        return json.dumps([item.toJSON() for item in self.images])

def evaluate():
    def index(score):
        ind=0
        if score>40 and score<=80:
            ind=1
        if score>80 and score<=90:
            ind=2
        if score>90:
            ind=3
        return ind
    path1 = "/home/qianqianjun/桌面/编辑结果"
    path2 = "/home/qianqianjun/桌面/AttGAN"
    path3 = "/home/qianqianjun/桌面/StarGAN"
    path="/home/qianqianjun/桌面/原图"
    request_url = "https://aip.baidubce.com/rest/2.0/face/v3/match"
    access_token = "24.7e8b426b9641416edee23272ea75dd2f.2592000.1588578630.282335-19253731"
    request_url = "{}?access_token={}".format(request_url, access_token)
    headers = {'content-type': 'application/json'}
    rank1=[0,0,0,0,0]
    rank2=[0,0,0,0,0]
    rank3=[0,0,0,0,0]
    scores1=[]
    scores2=[]
    scores3=[]
    names = os.listdir(path1)
    for name in names:
        data1=Data(os.path.join(path,name),os.path.join(path1,name))
        data2=Data(os.path.join(path,name),os.path.join(path2,name))
        data3=Data(os.path.join(path,name),os.path.join(path3,name))
        params1 = data1.toJSON()
        params2=data2.toJSON()
        params3=data3.toJSON()
        response1 = requests.post(request_url, data=params1, headers=headers)
        if response1:
            try:
                score1 = response1.json()["result"]["score"]
                rank1[index(score1)] += 1
                scores1.append(score1)
            except:
                print(response1.json())
                rank1[-1]+=1
        time.sleep(0.6)
        response2=requests.post(request_url,data=params2,headers=headers)
        if response2:
            try:
                score2=response2.json()["result"]["score"]
                rank2[index(score2)] += 1
                scores2.append(score2)
            except:
                print(response2.json())
                rank2[-1]+=1
        time.sleep(0.6)
        response3 = requests.post(request_url, data=params3, headers=headers)
        if response3:
            try:
                score3 = response3.json()["result"]["score"]
                rank3[index(score3)] += 1
                scores3.append(score3)
            except:
                print(response3.json())
                rank3[-1] += 1
        time.sleep(0.6)
    with open("rank1","wb") as f:
        pk.dump(rank1,f)
    with open("rank2","wb") as f:
        pk.dump(rank2,f)
    with open("rank3","wb") as f:
        pk.dump(rank3,f)
    with open("scores1","wb") as f:
        pk.dump(scores1,f)
    with open("scores2","wb") as f:
        pk.dump(scores2,f)
    with open("scores3","wb") as f:
        pk.dump(scores3,f)
def showRect():
    with open("rank1","rb") as f:
        rank1=pk.load(f,encoding="utf-8")
        data1=[rank1[-1]]+ rank1[:-1]
    with open("rank2","rb") as f:
        rank2=pk.load(f,encoding="utf-8")
        data2=[rank2[-1]]+ rank2[:-1]
    with open("rank3","rb") as f:
        rank3=pk.load(f,encoding="utf-8")
        data3=[rank3[-1]]+ rank3[:-1]
    x=range(len(data1))
    label_list=['找不到人脸','可能性极低', '可能性较低', '可能性较高', '可能性极高']
    rects1=plt.bar(x,height=data1,width=0.4,alpha=0.8,color="red",label="Ours")
    rects2=plt.bar([i+0.4 for i in x],width=0.4,height=data2,color="blue",label="AttGAN")
    rects3 = plt.bar([i + 0.8 for i in x], width=0.4, height=data3, color="blue", label="StarGAN")
    plt.ylabel("图片数量")
    plt.xticks([index + 0.2 for index in x], label_list)
    plt.xlabel("编辑结果图像与原图同一人可能性级别")
    plt.title("重建图像质量评估")
    plt.legend()
    for rect in rects1:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    for rect in rects3:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height + 1, str(height), ha="center", va="bottom")
    plt.savefig(fname="rects.png",figsize=(10,10))
    plt.show()
def showHist():
    with open("scores1", "rb") as f:
        scores1 = pk.load(f, encoding="utf-8")
    with open("scores2", "rb") as f:
        scores2 = pk.load(f, encoding="utf-8")
    with open("scores3", "rb") as f:
        scores3 = pk.load(f, encoding="utf-8")
    plt.figure(figsize=(8,4))
    plt.subplot(1,3,1)
    plt.hist(scores1, bins=40, density=0, facecolor="red", edgecolor="black", alpha=0.7)
    plt.xlabel("相似度区间%")
    plt.ylabel("相似度频率")
    plt.title("ours编辑后相似度分布直方图")
    plt.subplot(1,3,2)
    plt.hist(scores2,bins=40, density=0, facecolor="blue", edgecolor="black", alpha=0.7)
    plt.xlabel("相似度区间%")
    plt.title("AttGAN编辑后相似度分布直方图")
    plt.subplot(1, 3, 3)
    plt.hist(scores3, bins=40, density=0, facecolor="green", edgecolor="black", alpha=0.7)
    plt.xlabel("相似度区间%")
    plt.title("StarGAN编辑后相似度分布直方图")
    plt.savefig(fname="hist.png",figsize=(8,4))
    plt.show()

"""
注意，要先调用evaluate函数完成评估，这是一个漫长的过程，每秒仅可以处理两张图像。
完成评估后可以绘制质量评估结果图和相似度直方图
"""
if __name__ == '__main__':
    # 使用百度api得到人脸相似度
    # evaluate()
    # 绘制质量评估图
    showRect()
    # 绘制相似度直方图
    showHist()