import lpips
import glob
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms

parser = argparse.ArgumentParser(description='Train the MaskNet on target masks')
parser.add_argument('--output', '-f', type=str, default="output", help='Load model from a .pth file')
parser.add_argument('--gt', '-s', type=str, default="gt", help='Downscaling factor of the images')
parser.add_argument('--txt', type=str, default="gt", help='Downscaling factor of the images')
args = parser.parse_args()
transform = transforms.Compose([
				transforms.ToTensor()])
# output_imgs = args.output+"*.png"
# gt_imgs = args.gt+"/*.png"
# print(output_imgs,gt_imgs)
# output_list = glob.glob(output_imgs)
# output_list.sort()
# gt_list = glob.glob(gt_imgs)
# gt_list.sort()
lpips_sum = 0
output_imgs = args.output+"/*.jpg"
gt_imgs = args.gt+"/*.jpg"
output_list = glob.glob(output_imgs)
gt_list = glob.glob(gt_imgs)
print(len(gt_list),len(output_list))

# print(output_list,gt_list)
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
# loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization

def transform_img(img_path,transform):    
    img = Image.open(img_path).convert('RGB')
    img=transform(img)
    img = (img-0.5)*2
    return img

for pred_path,gt_path in zip(output_list,gt_list):
    # print(pred_path,gt_path)
    pred_img = transform_img(pred_path,transform)
    gt_img = transform_img(gt_path,transform)  

    d = loss_fn_alex(pred_img, gt_img)
    lpips_sum += d
lpips_score = lpips_sum/len(gt_list)
with open("{}".format(args.txt),mode="a") as f:
    f.write("LPIPS:{}".format(lpips_score.item()))
    f.write("\n")
print("\n\n")
print(args.output)
print(lpips_score)
print("\n\n")