from PIL import Image
import os
import glob

def resize_images_batch(input_folder, output_folder, target_size=(512, 512)):
    """
    批量将图片调整为指定尺寸
    
    参数:
    input_folder: 输入图片文件夹路径
    output_folder: 输出图片文件夹路径
    target_size: 目标尺寸，默认为(512, 512)
    """
    
    # 创建输出文件夹（如果不存在）
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 支持的图片格式
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif', '*.tiff']
    
    # 获取所有图片文件
    image_files = []
    for extension in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_folder, extension)))
        image_files.extend(glob.glob(os.path.join(input_folder, extension.upper())))
    
    if not image_files:
        print(f"在 {input_folder} 中没有找到图片文件")
        return
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    success_count = 0
    error_count = 0
    
    for i, image_path in enumerate(image_files):
        try:
            # 获取文件名（不包括路径）
            filename = os.path.basename(image_path)
            
            # 打开图片
            with Image.open(image_path) as img:
                # 获取原始尺寸
                original_size = img.size
                
                # 调整图片大小（使用LANCZOS重采样算法，质量较好）
                resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # 构建输出路径
                output_path = os.path.join(output_folder, filename)
                
                # 保存调整后的图片
                resized_img.save(output_path, quality=95, optimize=True)
                
                print(f"[{i+1}/{len(image_files)}] {filename}: {original_size} -> {target_size}")
                success_count += 1
                
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
            error_count += 1
    
    print(f"\n处理完成！")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")

def resize_single_image(input_path, output_path, target_size=(512, 512)):
    """
    调整单个图片的尺寸
    
    参数:
    input_path: 输入图片路径
    output_path: 输出图片路径
    target_size: 目标尺寸
    """
    try:
        with Image.open(input_path) as img:
            original_size = img.size
            resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
            resized_img.save(output_path, quality=95, optimize=True)
            print(f"图片已调整: {original_size} -> {target_size}")
            return True
    except Exception as e:
        print(f"处理图片时出错: {str(e)}")
        return False

# 使用示例
if __name__ == "__main__":
    # 方法1: 批量处理整个文件夹
    input_folder = "/home/xyhugo/2D-SpaceEdit/preprocess/photos"      # 替换为你的输入文件夹路径
    output_folder = "/home/xyhugo/2D-SpaceEdit/preprocess/512_photos"  # 替换为你的输出文件夹路径
    
    resize_images_batch(input_folder, output_folder, (512, 512))
    
    # 方法2: 处理单个图片
    # resize_single_image("input.jpg", "output.jpg", (512, 512))