"""
定量分析实验
用于检测人脸是否具有某一种属性以及属性的置信度
"""
import os
import numpy as np
import json
import requests
import time
import base64
from quantitative import image,Data
class Evaluate(object):
    def __init__(self):
        self.request_url = "https://aip.baidubce.com/rest/2.0/face/v3/detect"
        self.access_token = "24.808e82fcacc20813dc4d113d742f72ca.2592000.1592145086.282335-19253731"
        self.request_url = "{}?access_token={}".format(request_url, access_token)
        self.headers = {'content-type': 'application/json'}
    def _run(self,path:str,save_name:str):
        """
        使用百度AI接口进行人脸检测，获得人脸是否具有某一种属性，以及具有该属性的置信度
        :param path:  待检测图片的目录
        :param save_name:  检测结果的保存名称
        :return:
        """
        paths = [os.path.join(path, name) for name in os.listdir(path)]
        returns = []
        for dirs in paths:
            with open(dirs, "rb") as f:
                encoder = base64.b64encode(f.read())
                text = encoder.decode()
            img = image(text)
            params = img.toJSON()
            response = requests.post(self.request_url, data=params, headers=self.headers)
            if response:
                try:
                    result = response.json()["result"]["face_list"][0]
                    expression_type = result["expression"]["type"]
                    exp_probability = result["expression"]["probability"]
                    emotion = result["emotion"]["type"]
                    emo_probability = result["emotion"]["probability"]
                    returns.append(
                        "{} {} {} {}".format(expression_type, exp_probability, emotion, emo_probability))
                except:
                    print(response.json())
            time.sleep(0.6)
        # 保存结果
        with open("./{}".format(save_name), "wb") as f:
            pk.dump(returns, f)
    def evalute(self,path,save_name):
        """
        使用百度AI人脸检测接口对某一个文件夹下的图像进行人脸检测
        :param path:  图像文件夹
        :param save_name: 结果保存的名称
        :return: 无返回值，结果会被保存
        """
        if isinstance(path,str):
            self._run(path,save_name=save_name)
        else:
            assert isinstance(path,list)
            assert isinstance(save_name,list)
            for dir,name in zip(path,save_name):
                self._run(dir,save_name=name)


if __name__ == '__main__':
    Op=Evaluate()
    images_path="/home/qianqianjun/桌面/编辑结果"
    results_path="/home/qianqianjun/桌面/检测结果"
    attributes=os.listdir(path)
    paths=[os.path.join(images_path,attribute) for attribute in attributes]
    names=[os.path.join(results_path,attribute) for attribute in attributes]
    Op.evalute(paths,names)