import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor
from PIL import Image
import os

class SimplifiedSAMProcessor:
    def __init__(self, model_type="vit_h", checkpoint_path="/home/xyhugo/3d-scene-editor/checkpoints/sam/sam_vit_h_4b8939.pth", max_display_size=800, output_base_dir="/home/xyhugo/2D-SpaceEdit/outputs"):
        """初始化SAM模型"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # 加载模型
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=self.device)
        
        # 创建预测器
        self.predictor = SamPredictor(self.sam)
        
        # 存储当前状态
        self.original_image = None
        self.current_mask = None
        self.points = []
        self.labels = []
        
        # 涂抹模式相关
        self.paint_mode = False  # 是否处于涂抹模式
        self.is_painting = False  # 是否正在涂抹
        self.paint_mask = None  # 涂抹的mask
        self.brush_size = 20  # 画笔大小

        # 新增：阶段累计mask
        self.combined_mask = None  # 已累计的阶段性mask（并集）
        
        # 显示相关
        self.max_display_size = max_display_size
        self.scale_factor = 1.0
        self.display_image = None
        
        # 新增：mask扩张参数
        self.dilation_kernel_size = 15  # 默认扩张核大小
        self.dilation_iterations = 1    # 扩张迭代次数
        
        # 输出目录配置
        self.output_base_dir = output_base_dir
        self.output_dirs = {
            'objects': os.path.join(output_base_dir, 'objects'),
            'holes': os.path.join(output_base_dir, 'holes'),
            'mask_dilated': os.path.join(output_base_dir, 'mask_dilated'),
            'mask_precise': os.path.join(output_base_dir, 'mask_precise'),
            'mask_edge': os.path.join(output_base_dir, 'mask_edge')  # 新增：扩张边缘mask目录
        }

        # 记录最近一次保存的结果，便于上层流程读取
        self.last_save_result = None

        # 创建输出目录
        self.create_output_directories()
        
    def create_output_directories(self):
        """创建输出目录"""
        for dir_name, dir_path in self.output_dirs.items():
            if not os.path.exists(dir_path):
                os.makedirs(dir_path, exist_ok=True)
                print(f"✓ 创建目录: {dir_path}")
            else:
                print(f"✓ 目录已存在: {dir_path}")
    
    def resize_for_display(self, image):
        """调整图像到合适的显示尺寸"""
        h, w = image.shape[:2]
        
        if max(h, w) <= self.max_display_size:
            self.scale_factor = 1.0
            return image
        
        self.scale_factor = self.max_display_size / max(h, w)
        new_w = int(w * self.scale_factor)
        new_h = int(h * self.scale_factor)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"图像缩放: {w}x{h} -> {new_w}x{new_h} (缩放比例: {self.scale_factor:.2f})")
        
        return resized
    
    def display_coord_to_original(self, x, y):
        """将显示坐标转换为原图坐标"""
        return int(x / self.scale_factor), int(y / self.scale_factor)
    
    def original_coord_to_display(self, x, y):
        """将原图坐标转换为显示坐标"""
        return int(x * self.scale_factor), int(y * self.scale_factor)
    
    def dilate_mask(self, mask, kernel_size=None, iterations=None):
        """
        扩张mask以改善inpainting效果
        
        Args:
            mask: 原始mask (bool array)
            kernel_size: 扩张核大小
            iterations: 迭代次数
            
        Returns:
            dilated_mask: 扩张后的mask
        """
        if kernel_size is None:
            kernel_size = self.dilation_kernel_size
        if iterations is None:
            iterations = self.dilation_iterations
            
        # 转换为uint8格式
        mask_uint8 = mask.astype(np.uint8) * 255
        
        # 创建扩张核
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        
        # 执行形态学扩张
        dilated_uint8 = cv2.dilate(mask_uint8, kernel, iterations=iterations)
        
        # 转换回bool
        dilated_mask = dilated_uint8 > 128
        
        return dilated_mask
    
    def get_edge_mask(self, original_mask, kernel_size=None, iterations=None):
        """
        获取扩张边缘部分的mask（扩张区域 - 原始区域）
        
        Args:
            original_mask: 原始mask (bool array)
            kernel_size: 扩张核大小
            iterations: 迭代次数
            
        Returns:
            edge_mask: 边缘部分的mask
        """
        dilated_mask = self.dilate_mask(original_mask, kernel_size, iterations)
        edge_mask = dilated_mask & (~original_mask)
        return edge_mask
    
    def adjust_dilation_size(self, change):
        """调整扩张核大小"""
        self.dilation_kernel_size = max(1, self.dilation_kernel_size + change)
        print(f"📏 扩张核大小: {self.dilation_kernel_size}")
    
    def adjust_brush_size(self, change):
        """调整画笔大小"""
        self.brush_size = max(5, min(100, self.brush_size + change))
        print(f"🖌️ 画笔大小: {self.brush_size}")

    # 新增：获取当前可提交的工作mask（优先涂抹中的mask）
    def get_current_working_mask(self):
        if self.paint_mode and self.paint_mask is not None and np.any(self.paint_mask):
            return self.paint_mask
        return self.current_mask

    # 新增：提交当前阶段到combined_mask
    def commit_stage(self):
        """
        将当前工作mask并入阶段累计combined_mask，并清空当前选择，方便继续下一阶段。
        支持直接提交涂抹中的mask（无需先按 f 确认）。
        """
        mask = self.get_current_working_mask()
        if mask is None or not np.any(mask):
            print("❌ 没有可提交的区域，请先分割或涂抹一些区域。")
            return False

        if self.combined_mask is None:
            self.combined_mask = mask.copy()
        else:
            self.combined_mask = np.logical_or(self.combined_mask, mask)

        area = int(np.sum(mask))
        total = mask.shape[0] * mask.shape[1]
        pct = area / total * 100
        combined_area = int(np.sum(self.combined_mask))
        combined_pct = combined_area / total * 100
        print(f"✅ 已提交阶段区域: {area} 像素 ({pct:.2f}%)")
        print(f"📦 累计并集面积: {combined_area} 像素 ({combined_pct:.2f}%)")

        # 清空当前工作选择，方便下一次分割
        self.points = []
        self.labels = []
        self.current_mask = None
        if self.paint_mask is not None:
            self.paint_mask.fill(False)

        self.update_display()
        return True

    # 新增：清空阶段累计
    def clear_combined(self):
        self.combined_mask = None
        print("🗑️ 已清空阶段累计区域。")
        self.update_display()
    
    def toggle_paint_mode(self):
        """切换涂抹模式"""
        self.paint_mode = not self.paint_mode
        if self.paint_mode:
            print(f"🎨 进入涂抹模式 (画笔大小: {self.brush_size})")
            print("   左键涂抹，右键清除，ESC退出涂抹模式")
            # 初始化涂抹mask
            if self.original_image is not None:
                h, w = self.original_image.shape[:2]
                self.paint_mask = np.zeros((h, w), dtype=bool)
        else:
            print("🎯 退出涂抹模式，回到SAM点击模式")
        self.update_display()
    
    def paint_at_position(self, x, y):
        """在指定位置涂抹"""
        if self.paint_mask is None:
            return
        
        # 转换为原图坐标
        orig_x, orig_y = self.display_coord_to_original(x, y)
        
        # 计算画笔在原图中的大小
        brush_radius = int(self.brush_size / self.scale_factor / 2)
        brush_radius = max(1, brush_radius)
        
        # 获取原图尺寸
        h, w = self.paint_mask.shape
        
        # 创建圆形画笔
        y_coords, x_coords = np.ogrid[:h, :w]
        mask_circle = (x_coords - orig_x)**2 + (y_coords - orig_y)**2 <= brush_radius**2
        
        # 涂抹到mask上
        self.paint_mask |= mask_circle
        
        # 更新显示
        self.update_display()
    
    def clear_paint_mask(self):
        """清除涂抹mask"""
        if self.paint_mask is not None:
            self.paint_mask.fill(False)
            print("🧹 清除涂抹区域")
            self.update_display()
    
    def finalize_paint_mask(self):
        """将涂抹mask转为当前mask"""
        if self.paint_mask is not None and np.any(self.paint_mask):
            self.current_mask = self.paint_mask.copy()
            # 清除点击点，因为现在用的是涂抹mask
            self.points = []
            self.labels = []
            print("✅ 涂抹区域已设为当前mask")
            return True
        return False
        
    def mouse_click(self, event, x, y, flags, param):
        """鼠标点击回调函数"""
        if self.paint_mode:
            # 涂抹模式
            if event == cv2.EVENT_LBUTTONDOWN:
                self.is_painting = True
                self.paint_at_position(x, y)
            elif event == cv2.EVENT_MOUSEMOVE and self.is_painting:
                self.paint_at_position(x, y)
            elif event == cv2.EVENT_LBUTTONUP:
                self.is_painting = False
            elif event == cv2.EVENT_RBUTTONDOWN:
                # 右键清除涂抹
                self.clear_paint_mask()
        else:
            # SAM点击模式
            if event == cv2.EVENT_LBUTTONDOWN:
                orig_x, orig_y = self.display_coord_to_original(x, y)
                self.points.append([orig_x, orig_y])
                self.labels.append(1)
                print(f"添加正向点: ({orig_x}, {orig_y})")
                self.update_mask()
                
            elif event == cv2.EVENT_RBUTTONDOWN:
                orig_x, orig_y = self.display_coord_to_original(x, y)
                self.points.append([orig_x, orig_y])
                self.labels.append(0)
                print(f"添加负向点: ({orig_x}, {orig_y})")
                self.update_mask()
    
    def update_mask(self):
        """更新并显示mask（SAM模式）"""
        if len(self.points) == 0:
            return
            
        points = np.array(self.points)
        labels = np.array(self.labels)
        
        masks, scores, _ = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=True,
        )
        
        # 选择得分最高的mask
        best_mask = masks[np.argmax(scores)]
        self.current_mask = best_mask
        
        self.update_display()
        
        # 显示选择区域统计
        area = np.sum(best_mask)
        dilated_mask = self.dilate_mask(best_mask)
        dilated_area = np.sum(dilated_mask)
        edge_area = np.sum(self.get_edge_mask(best_mask))
        total_area = best_mask.shape[0] * best_mask.shape[1]
        percentage = (area / total_area) * 100
        dilated_percentage = (dilated_area / total_area) * 100
        edge_percentage = (edge_area / total_area) * 100
        print(f"原始区域: {area} 像素 ({percentage:.1f}%)")
        print(f"扩张区域: {dilated_area} 像素 ({dilated_percentage:.1f}%)")
        print(f"边缘区域: {edge_area} 像素 ({edge_percentage:.1f}%)")
        print(f"扩张核大小: {self.dilation_kernel_size}")
    
    def update_display(self):
        """更新显示（通用方法）"""
        if self.display_image is None:
            return
        
        display_image = self.display_image.copy()
        
        # 确定要显示的mask
        mask_to_show = None
        display_mask = None  # 修复：先初始化，避免未赋值引用
        if self.paint_mode and self.paint_mask is not None:
            mask_to_show = self.paint_mask
        elif self.current_mask is not None:
            mask_to_show = self.current_mask
        
        if mask_to_show is not None:
            # 调整mask到显示尺寸
            if self.scale_factor != 1.0:
                display_mask = cv2.resize(
                    mask_to_show.astype(np.uint8), 
                    (self.display_image.shape[1], self.display_image.shape[0]), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                display_mask = mask_to_show
            
            # 显示原始mask（绿色半透明）
            mask_overlay = display_image.copy()
            mask_overlay[display_mask] = [0, 255, 0]
            display_image = cv2.addWeighted(display_image, 0.7, mask_overlay, 0.3, 0)
            
            # 如果不是涂抹模式，显示扩张边缘
            if not self.paint_mode:
                dilated_mask = self.dilate_mask(mask_to_show)
                if self.scale_factor != 1.0:
                    dilated_display = cv2.resize(
                        dilated_mask.astype(np.uint8), 
                        (self.display_image.shape[1], self.display_image.shape[0]), 
                        interpolation=cv2.INTER_NEAREST
                    ).astype(bool)
                else:
                    dilated_display = dilated_mask
                    
                # 只显示扩张边缘（红色）
                edge_mask = dilated_display & (~display_mask)
                edge_overlay = display_image.copy()
                edge_overlay[edge_mask] = [0, 0, 255]
                display_image = cv2.addWeighted(display_image, 0.8, edge_overlay, 0.2, 0)

        # 新增：显示已累计的阶段mask（蓝色半透明，仅显示未与当前mask重叠的部分）
        if self.combined_mask is not None:
            if self.scale_factor != 1.0:
                combined_display = cv2.resize(
                    self.combined_mask.astype(np.uint8),
                    (self.display_image.shape[1], self.display_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                combined_display = self.combined_mask

            combined_only = combined_display
            if display_mask is not None:
                combined_only = combined_display & (~display_mask)

            if np.any(combined_only):
                combined_overlay = display_image.copy()
                combined_overlay[combined_only] = [255, 0, 0]  # 蓝色（BGR）
                display_image = cv2.addWeighted(display_image, 0.85, combined_overlay, 0.15, 0)
        
        # 显示点击点（仅在SAM模式下）
        if not self.paint_mode:
            for point, label in zip(self.points, self.labels):
                disp_x, disp_y = self.original_coord_to_display(point[0], point[1])
                color = (0, 255, 0) if label == 1 else (0, 0, 255)
                cv2.circle(display_image, (disp_x, disp_y), 5, color, -1)
                cv2.circle(display_image, (disp_x, disp_y), 5, (255, 255, 255), 2)
        
        # 添加模式指示器
        mode_text = "🎨 涂抹模式" if self.paint_mode else "🎯 SAM模式"
        if self.paint_mode:
            mode_text += f" (画笔: {self.brush_size})"
        if self.combined_mask is not None:
            mode_text += " + 已累计"
        cv2.putText(display_image, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_image, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        cv2.imshow('SAM Object Selector', display_image)
    
    def save_transparent_object(self, base_name):
        """保存透明背景的选中物体（使用原始精确mask）"""
        # 使用最终mask（combined优先）
        final_mask = self.combined_mask if self.combined_mask is not None else self.current_mask
        if final_mask is None:
            print("❌ 请先选择或累计物体区域！")
            return None
        
        mask = final_mask
        image = self.original_image
        
        # 找到物体边界
        y_indices, x_indices = np.where(mask)
        if len(x_indices) == 0:
            print("❌ 没有选中任何区域！")
            return None
        
        x_min, x_max = x_indices.min(), x_indices.max()
        y_min, y_max = y_indices.min(), y_indices.max()
        
        # 裁剪物体区域
        cropped_image = image[y_min:y_max+1, x_min:x_max+1]
        cropped_mask = mask[y_min:y_max+1, x_min:x_max+1]
        
        # 创建RGBA图像（透明背景）
        h, w = cropped_image.shape[:2]
        result = np.zeros((h, w, 4), dtype=np.uint8)
        result[:, :, :3] = cropped_image
        result[:, :, 3] = cropped_mask.astype(np.uint8) * 255
        
        # 保存到objects目录
        object_path = os.path.join(self.output_dirs['objects'], f"{base_name}_object.png")
        pil_image = Image.fromarray(result, 'RGBA')
        pil_image.save(object_path)
        
        print(f"✓ 透明背景物体: {object_path} ({w}x{h})")
        return object_path
    
    def save_background_with_hole(self, base_name, use_dilated_mask=True):
        """
        保存去掉物体后的背景（有洞）
        
        Args:
            base_name: 文件名前缀
            use_dilated_mask: 是否使用扩张后的mask
        """
        # 使用最终mask（combined优先）
        final_mask = self.combined_mask if self.combined_mask is not None else self.current_mask
        if final_mask is None:
            print("❌ 请先选择或累计物体区域！")
            return None
        
        # 选择使用原始mask还是扩张mask
        if use_dilated_mask:
            mask = self.dilate_mask(final_mask)
            suffix = "_background_dilated_hole"
            print(f"🔍 使用扩张mask (核大小: {self.dilation_kernel_size})")
        else:
            mask = final_mask
            suffix = "_background_hole"
            print("🔍 使用原始精确mask")
        
        image = self.original_image
        
        # 创建有洞的背景
        background = image.copy()
        
        # 将选中区域设为黑色（表示洞）
        background[mask] = [0, 0, 0]
        
        # 保存到holes目录
        background_path = os.path.join(self.output_dirs['holes'], f"{base_name}{suffix}.jpg")
        pil_image = Image.fromarray(background)
        pil_image.save(background_path, quality=95)
        
        print(f"✓ 有洞背景: {background_path}")
        return background_path
    
    def save_mask_file(self, base_name, use_dilated_mask=True):
        """
        保存mask文件（给PixelHacker使用）
        
        Args:
            base_name: 文件名前缀
            use_dilated_mask: 是否使用扩张后的mask
        """
        # 使用最终mask（combined优先）
        final_mask = self.combined_mask if self.combined_mask is not None else self.current_mask
        if final_mask is None:
            print("❌ 请先选择或累计物体区域！")
            return None
        
        if use_dilated_mask:
            mask = self.dilate_mask(final_mask)
            suffix = "_mask"
            output_dir = self.output_dirs['mask_dilated']
            print(f"🔍 保存扩张mask (核大小: {self.dilation_kernel_size})")
        else:
            mask = final_mask
            suffix = "_mask"
            output_dir = self.output_dirs['mask_precise']
            print("🔍 保存原始精确mask")
        
        # 保存mask为PNG
        mask_uint8 = mask.astype(np.uint8) * 255
        mask_path = os.path.join(output_dir, f"{base_name}{suffix}.png")
        cv2.imwrite(mask_path, mask_uint8)
        
        print(f"✓ Mask文件: {mask_path}")
        return mask_path
    
    def save_edge_mask_file(self, base_name):
        """
        保存扩张边缘mask文件（仅包含扩张部分）
        
        Args:
            base_name: 文件名前缀
        """
        # 使用最终mask（combined优先）
        final_mask = self.combined_mask if self.combined_mask is not None else self.current_mask
        if final_mask is None:
            print("❌ 请先选择或累计物体区域！")
            return None
        
        edge_mask = self.get_edge_mask(final_mask)
        
        # 保存边缘mask为PNG
        mask_uint8 = edge_mask.astype(np.uint8) * 255
        edge_path = os.path.join(self.output_dirs['mask_edge'], f"{base_name}_edge_mask.png")
        cv2.imwrite(edge_path, mask_uint8)
        
        edge_area = np.sum(edge_mask)
        total_area = edge_mask.shape[0] * edge_mask.shape[1]
        edge_percentage = (edge_area / total_area) * 100

        print(f"✓ 边缘Mask文件: {edge_path}")
        print(f"🔍 边缘区域: {edge_area} 像素 ({edge_percentage:.1f}%) (核大小: {self.dilation_kernel_size})")
        return edge_path
    
    def save_all_outputs(self, base_name, use_dilated_for_inpainting=True):
        """
        保存所有输出文件到对应目录
        
        Args:
            base_name: 文件名前缀
            use_dilated_for_inpainting: inpainting相关文件是否使用扩张mask
        """
        # 修复：允许仅使用 combined_mask 的情况
        final_mask = self.combined_mask if self.combined_mask is not None else self.current_mask
        if final_mask is None or not np.any(final_mask):
            print("❌ 请先选择或累计物体区域！")
            return None

        print(f"\n保存所有输出文件到 {self.output_base_dir}")
        print("-" * 60)
        
        # 保存透明背景物体到objects目录（始终使用精确mask）
        object_path = self.save_transparent_object(base_name)
        
        # 保存背景到holes目录（可选择是否使用扩张mask）
        background_path = self.save_background_with_hole(base_name, use_dilated_for_inpainting)
        
        # 保存mask文件到对应目录
        if use_dilated_for_inpainting:
            mask_path = self.save_mask_file(base_name, True)  # 扩张mask到mask_dilated目录
        else:
            mask_path = self.save_mask_file(base_name, False)  # 精确mask到mask_precise目录
        
        # 同时保存另一种mask以备后用
        if use_dilated_for_inpainting:
            precise_mask_path = self.save_mask_file(base_name, False)  # 精确mask
        else:
            precise_mask_path = self.save_mask_file(base_name, True)   # 扩张mask
        
        # 保存边缘mask
        edge_mask_path = self.save_edge_mask_file(base_name)
        
        print("-" * 60)
        print("✅ 所有文件已保存到对应目录！")
        print(f"📁 输出目录结构:")
        print(f"   {self.output_dirs['objects']}")
        print(f"   {self.output_dirs['holes']}")
        print(f"   {self.output_dirs['mask_dilated']}")
        print(f"   {self.output_dirs['mask_precise']}")
        print(f"   {self.output_dirs['mask_edge']}")
        
        self.last_save_result = {
            'object': object_path,
            'background_hole': background_path,
            'mask': mask_path,
            'alternative_mask': precise_mask_path,
            'edge_mask': edge_mask_path
        }

        return self.last_save_result
    
    def process_image(self, image_path, auto_exit_on_save: bool = False):
        """处理图像并根据需要自动退出窗口"""
        # 重置最新结果
        self.last_save_result = None

        # 读取图像
        self.original_image = cv2.imread(image_path)
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        
        print(f"✓ 图像加载: {self.original_image.shape[1]}x{self.original_image.shape[0]}")
        
        # 设置SAM图像
        self.predictor.set_image(self.original_image)
        
        # 准备显示
        display_rgb = self.resize_for_display(self.original_image)
        self.display_image = cv2.cvtColor(display_rgb, cv2.COLOR_RGB2BGR)
        
        # 创建窗口
        cv2.namedWindow('SAM Object Selector', cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('SAM Object Selector', self.mouse_click)
        self.update_display()
        
        print("\n" + "="*70)
        print("🎯 SAM 物体选择器 (支持Mask扩张优化+阶段累计)")
        print("="*70)
        print("📁 输出目录配置:")
        print(f"   物体文件: {self.output_dirs['objects']}")
        print(f"   背景文件: {self.output_dirs['holes']}")
        print(f"   扩张Mask: {self.output_dirs['mask_dilated']}")
        print(f"   精确Mask: {self.output_dirs['mask_precise']}")
        print(f"   边缘Mask: {self.output_dirs['mask_edge']}")
        print()
        print("🖱️  鼠标操作:")
        print("   左键: 选择物体区域")
        print("   右键: 排除区域")
        print("\n⌨️  键盘操作:")
        print("   'SPACE': 💾 保存所有输出 (默认用扩张mask)")
        print("   'ENTER': 💾 保存所有输出 (用精确mask)")
        print("   'o':     💾 只保存透明背景物体")
        print("   'b':     💾 保存背景 (扩张mask)")
        print("   'B':     💾 保存背景 (精确mask)")
        print("   'e':     💾 只保存边缘mask")
        print("   'p':     🎨 切换涂抹模式")
        print("   'ESC':   🎯 退出涂抹模式")
        print("   'f':     ✅ 确认涂抹区域 (涂抹模式下)")
        print("   '[':     🖌️ 缩小画笔 (-5)")
        print("   ']':     🖌️ 放大画笔 (+5)")
        print("   '+':     🔍 增大扩张核 (+2)")
        print("   '-':     🔍 减小扩张核 (-2)")
        print("   'c':     📌 提交当前阶段到累计（组合并集）")
        print("   'C':     🗑️ 清空阶段累计")
        print("   'r':     🔄 重置当前选择（不影响已累计）")
        print("   'q':     ❌ 退出")
        print("="*70)
        print("💡 绿色=当前选择, 蓝色=已累计阶段区域, 红色边缘=当前选择的扩张区域")
        
        # 主循环
        should_exit = False

        while True:
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("👋 退出程序")
                break
            elif key == ord(' '):  # 空格键 - 保存所有输出（扩张mask）
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                results = self.save_all_outputs(base_name, use_dilated_for_inpainting=True)
                if results:
                    print(f"\n🎉 输出文件已保存 (使用扩张mask):")
                    print(f"   📦 物体文件: {os.path.basename(results['object'])}")
                    print(f"   🕳️  背景文件: {os.path.basename(results['background_hole'])}")
                    print(f"   🎭 主要Mask: {os.path.basename(results['mask'])}")
                    print(f"   🎭 备用Mask: {os.path.basename(results['alternative_mask'])}")
                    print(f"   🔲 边缘Mask: {os.path.basename(results['edge_mask'])}")
                    print(f"\n📁 文件位置:")
                    print(f"   {self.output_base_dir}/")
                    print(f"   ├── objects/")
                    print(f"   ├── holes/")
                    print(f"   ├── mask_dilated/")
                    print(f"   ├── mask_precise/")
                    print(f"   └── mask_edge/")
                    print(f"\n💡 下一步: 使用PixelHacker修复背景")
                    print(f"   推荐使用holes/目录中的扩张文件获得更好效果！")
                    if auto_exit_on_save:
                        should_exit = True
            elif key == 13:  # Enter键 - 保存所有输出（精确mask）
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                results = self.save_all_outputs(base_name, use_dilated_for_inpainting=False)
                if results:
                    print(f"\n🎉 输出文件已保存 (使用精确mask):")
                    print(f"   📦 物体文件: {os.path.basename(results['object'])}")
                    print(f"   🕳️  背景文件: {os.path.basename(results['background_hole'])}")
                    print(f"   🎭 主要Mask: {os.path.basename(results['mask'])}")
                    print(f"   🎭 备用Mask: {os.path.basename(results['alternative_mask'])}")
                    print(f"   🔲 边缘Mask: {os.path.basename(results['edge_mask'])}")
                    if auto_exit_on_save:
                        should_exit = True
            elif key == ord('o'):
                # 只保存物体
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                self.save_transparent_object(base_name)
            elif key == ord('b'):
                # 保存背景（扩张mask）
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                self.save_background_with_hole(base_name, use_dilated_mask=True)
            elif key == ord('B'):
                # 保存背景（精确mask）
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                self.save_background_with_hole(base_name, use_dilated_mask=False)
            elif key == ord('e'):
                # 只保存边缘mask
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                self.save_edge_mask_file(base_name)
            elif key == ord('+') or key == ord('='):
                # 增大扩张核
                self.adjust_dilation_size(2)
                if len(self.points) > 0:
                    self.update_mask()
            elif key == ord('-'):
                # 减小扩张核
                self.adjust_dilation_size(-2)
                if len(self.points) > 0:
                    self.update_mask()
            elif key == ord('p'):
                # 切换涂抹模式
                self.toggle_paint_mode()
            elif key == ord('c'):
                # 提交当前阶段
                self.commit_stage()
            elif key == ord('C'):
                # 清空阶段累计
                self.clear_combined()
            elif key == 27:  # ESC键
                # 退出涂抹模式
                if self.paint_mode:
                    self.paint_mode = False
                    print("🎯 退出涂抹模式，回到SAM点击模式")
                    self.update_display()
            elif key == ord('f'):
                # 确认涂抹区域
                if self.paint_mode and self.finalize_paint_mask():
                    self.update_display()
            elif key == ord('['):
                # 缩小画笔
                self.adjust_brush_size(-5)
            elif key == ord(']'):
                # 放大画笔
                self.adjust_brush_size(5)
            elif key == ord('r'):
                # 重置选择
                self.points = []
                self.labels = []
                self.current_mask = None
                if self.paint_mask is not None:
                    self.paint_mask.fill(False)
                self.update_display()
                print("🔄 已重置选择")

            if auto_exit_on_save and should_exit:
                print("✅ 已保存，自动退出SAM窗口")
                break
        
        cv2.destroyAllWindows()
        return self.last_save_result

def main():
    # 查找图像文件
    image_path = "/home/xyhugo/2D-SpaceEdit/materials/images/car1.png"
    
    if not os.path.exists(image_path):
        # 自动查找图像文件
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        current_images = [f for f in os.listdir('.') if f.lower().endswith(image_extensions)]
        
        if current_images:
            image_path = current_images[0]
            print(f"📸 自动选择图像: {image_path}")
        else:
            print("❌ 未找到图像文件！")
            print("请将图像文件放在当前目录下")
            return
    
    # 创建处理器并运行
    processor = SimplifiedSAMProcessor(max_display_size=800)
    processor.process_image(image_path)

if __name__ == "__main__":
    main()