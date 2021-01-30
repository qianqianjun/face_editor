import pickle
import PIL.Image
import numpy as np
import dnnlib.tflib as tflib
from structer.generator import Generator
import os
import shutil
def read_feature(path)->np.ndarray:
    """
    读取人脸潜在编码
    :param path: 编码文件所在目录
    :return: 潜在编码内容
    """
    with open(path, 'r') as f:
        contents=f.readlines()
        latent_code=np.array([float(line.strip("\n")) for line in contents],dtype=np.float32)
    return latent_code

def generate_image(latent_vector, generator)->PIL.Image:
    """
    :param latent_vector: 人脸潜在编码（经过编辑后的结果）
    :param generator: 生成器对象
    :return: 通过生成器和编辑后的潜在编码生成的图像
    """
    latent_vector = latent_vector.reshape((1, 18, 512))
    generator.set_dlatents(latent_vector)
    img_array = generator.generate_images()[0]
    img = PIL.Image.fromarray(img_array, 'RGB')
    return img

def move_latent_and_save(latent_vector, direction, coeffs, generator,save_path,attribute):
    """
    使用编辑之后的人脸潜在编码生成最终的人脸图像并保存
    :param latent_vector: 　人脸潜在编码（通过映射网络之后的结果）
    :param direction: 属性的向量表示
    :param coeffs:　权重（步长，程度）
    :param generator: 生成器对象
    :param save_path: 图像保存位置
    :param attribute: 属性名称，用于设置结果图像的保存目录
    :return: none
    """
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff*direction)[:8] # 经过测试，使用前8个（1*512）的向量可以保证生成结果稳定，同时效果也比较明显。
        result = generate_image(new_latent_vector, generator)
        result.save(os.path.join(save_path,"{}_{}.png".format(attribute,str(coeff))))
def main():
    tflib.init_tf() # 初始化系统参数并设置默认session
    with open('model/model.pkl', "rb") as f:
        generator_network, discriminator_network, Gs_network = pickle.load(f) # 加载训练好的styleGAN模型
    generator = Generator(Gs_network, batch_size=1, randomize_noise=False)
    test_dir="/home/qianqianjun/桌面/原图"
    result_dir="/home/qianqianjun/桌面/编辑结果"
    feature_codes=os.listdir(os.path.join(test_dir,"generate_code"))
    # 待测试的属性
    attributes=['smile','age','glasses','gender''emotion_angry','emotion_disgust','emotion_fear','emotion_sad','emotion_surprise']
    for attribute in attributes:
        for feature in feature_codes:
            img_name=feature.split(".")[0]
            save_path = os.path.join(result_dir, attribute, img_name)
            os.makedirs(save_path, exist_ok=True)
            # 加载潜在编码
            face_latent = read_feature(os.path.join(test_dir,"generate_code",feature))
            stack_latents = np.stack(face_latent for _ in range(1)) # (1,512)
            face_dlatent = Gs_network.components.mapping.run(stack_latents, None) #将潜在编码通过映射网络
            direction = np.load('attributes/{}.npy'.format(attribute)) #加载属性向量
            coeffs = [i*0.5 for i in range(-10,11)] # 这里表示效果的权重，每一个效果20个权重，10个正向权重，10个负向权重。
            shutil.copyfile(os.path.join(test_dir,"{}.png".format(img_name)),os.path.join(save_path,"origin_{}.png".format(img_name)))
            move_latent_and_save(face_dlatent, direction, coeffs, generator,save_path,attribute)
if __name__ == "__main__":
    main()