import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import importlib
import importlib.util
import threading
import queue
import subprocess

import dearpygui.dearpygui as dpg
import torch
from PIL import Image
import numpy as np

# Ensure repository root is on the path so relative imports resolve regardless of CWD
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

INPAINTING_DIR = REPO_ROOT / "inpainting"
TRELLIS_DIR = REPO_ROOT / "reconstruct_from_2d"

if str(INPAINTING_DIR) not in sys.path:
    sys.path.append(str(INPAINTING_DIR))

if str(TRELLIS_DIR) not in sys.path:
    sys.path.append(str(TRELLIS_DIR))


def _load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


pixelhacker_utils = _load_module("pixelhacker_utils", INPAINTING_DIR / "utils.py")
pixelhacker_pipeline_mod = _load_module("pixelhacker_pipeline", INPAINTING_DIR / "pipeline.py")
trellis_pipelines_mod = None
trellis_utils_mod = None

from preprocess.sam_processor import SimplifiedSAMProcessor  # noqa: E402
from diffusers import DDIMScheduler  # noqa: E402

build_model = pixelhacker_utils.build_model
build_vae = pixelhacker_utils.build_vae
load_cfg = pixelhacker_utils.load_cfg
PixelHacker_Pipeline = pixelhacker_pipeline_mod.PixelHacker_Pipeline


def _ensure_trellis_modules() -> None:
    global trellis_pipelines_mod, trellis_utils_mod
    if trellis_pipelines_mod is None:
        trellis_pipelines_mod = importlib.import_module("trellis.pipelines")
    if trellis_utils_mod is None:
        trellis_utils_mod = importlib.import_module("trellis.utils.postprocessing_utils")


class PixelHackerRunner:
    """Thin wrapper around PixelHacker pipeline for single-image inference."""

    def __init__(
        self,
        config_path: Path,
        weight_path: Path,
        device: str = "cuda",
        release_after_run: bool = False,
    ) -> None:
        requested_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_device = requested_device
        self.release_after_run = release_after_run
        self.config = load_cfg(str(config_path))

        # Build diffusion model and load weights
        self.model = build_model(self.config, 20).to("cpu")
        state_dict = torch.load(str(weight_path), map_location="cpu")
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[PixelHacker] Missing keys when loading weights: {missing}")
        if unexpected:
            print(f"[PixelHacker] Unexpected keys in weights: {unexpected}")
        self.model.eval()

        # Build VAE and scheduler
        self.vae = build_vae(self.config).to("cpu")
        self.vae.eval()
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )

        self.pipeline = PixelHacker_Pipeline(
            model=self.model,
            vae=self.vae,
            scheduler=self.scheduler,
            device="cpu",
            dtype=torch.float,
        )

        vae_downsample = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_size = self.model.diff_model.config.sample_size * vae_downsample

    def _prepare_for_run(self) -> None:
        device = self.target_device if torch.cuda.is_available() and self.target_device.type == "cuda" else torch.device("cpu")
        device_str = str(device)
        self.model.to(device)
        self.vae.to(device)
        self.pipeline.device = device_str
        self.pipeline.model.to(device)
        self.pipeline.vae.to(device)
        if hasattr(self.pipeline, "input_ids"):
            self.pipeline.input_ids = self.pipeline.input_ids.to(device)

    def release(self) -> None:
        self.model.to("cpu")
        self.vae.to("cpu")
        self.pipeline.device = "cpu"
        self.pipeline.model.to("cpu")
        self.pipeline.vae.to("cpu")
        if hasattr(self.pipeline, "input_ids"):
            self.pipeline.input_ids = self.pipeline.input_ids.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run(
        self,
        image_path: Path,
        mask_path: Path,
        output_path: Path,
        *,
        num_steps: int = 20,
        strength: float = 0.999,
        guidance_scale: float = 4.5,
        noise_offset: Optional[float] = 0.0357,
    ) -> Path:
        self._prepare_for_run()
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        result = self.pipeline(
            [image],
            [mask],
            image_size=self.image_size,
            num_steps=num_steps,
            strength=strength,
            guidance_scale=guidance_scale,
            noise_offset=noise_offset,
            paste=False,
            mute=False,
        )[0]

        output_path.parent.mkdir(parents=True, exist_ok=True)
        result.save(output_path)
        if self.release_after_run:
            self.release()
        return output_path


class TrellisRunner:
    """Wrapper around Trellis image-to-3D pipeline with persisted configuration."""

    def __init__(
        self,
        model_id: str,
        device: str,
        simplify_ratio: float,
        texture_size: int,
        default_seed: int,
        offline_mode: bool = False,
        max_retries: int = 2,
        retry_delay: float = 5.0,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.simplify_ratio = simplify_ratio
        self.texture_size = texture_size
        self.default_seed = default_seed
        self.offline_mode = offline_mode
        self.max_retries = max(1, max_retries)
        self.retry_delay = max(0.0, retry_delay)
        self.pipeline = None
        self._pipeline_cls = None

        if self.offline_mode:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("SPCONV_ALGO", "native")

    def _ensure_pipeline(self) -> None:
        if self.pipeline is not None:
            return
        _ensure_trellis_modules()
        if self._pipeline_cls is None:
            self._pipeline_cls = trellis_pipelines_mod.TrellisImageTo3DPipeline

        pipeline = self._pipeline_cls.from_pretrained(self.model_id)
        if self.device.startswith("cuda"):
            if hasattr(pipeline, "cuda"):
                pipeline.cuda()
            else:
                pipeline.to(self.device)
        else:
            pipeline.to(self.device)
        self.pipeline = pipeline

    def preload(self) -> None:
        self._ensure_pipeline()

    def release(self) -> None:
        if self.pipeline is None:
            return
        try:
            if hasattr(self.pipeline, "to"):
                self.pipeline.to("cpu")
            elif hasattr(self.pipeline, "cpu"):
                self.pipeline.cpu()
        except Exception:
            pass
        self.pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _is_transient_error(self, exc: Exception) -> bool:
        message = str(exc)
        transient_markers = (
            "Remote end closed connection",
            "Connection reset",
            "timed out",
            "Temporary failure",
        )
        return any(marker in message for marker in transient_markers)

    def run(
        self,
        image_path: Path,
        output_root: Path,
        seed: Optional[int] = None,
    ) -> Dict[str, Path]:
        attempts = 0
        outputs = None
        while attempts < self.max_retries:
            attempts += 1
            self._ensure_pipeline()
            if self.pipeline is None:
                raise RuntimeError("Trellis pipeline failed to load")

            try:
                pil_image = Image.open(image_path).convert("RGBA")
                run_seed = seed if seed is not None else self.default_seed
                outputs = self.pipeline.run(
                    pil_image,
                    seed=run_seed,
                    preprocess_image=False,
                    formats=["gaussian", "mesh"],
                )
                break
            except Exception as exc:  # pragma: no cover - runtime diagnostics
                transient = self._is_transient_error(exc)
                if attempts < self.max_retries and transient:
                    print(f"[Trellis] Generation failed (attempt {attempts}/{self.max_retries}). Retrying in {self.retry_delay}s...")
                    self.pipeline = None
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(self.retry_delay)
                    continue
                raise
        if outputs is None:
            raise RuntimeError("Trellis generation failed without producing outputs")

        gaussian = outputs["gaussian"][0]
        mesh = outputs["mesh"][0]

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = output_root / f"{image_path.stem}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)

        ply_path = run_dir / "model.ply"
        gaussian.save_ply(str(ply_path))

        _ensure_trellis_modules()
        glb_obj = trellis_utils_mod.to_glb(
            gaussian,
            mesh,
            simplify=self.simplify_ratio,
            texture_size=self.texture_size,
            verbose=False,
        )
        glb_path = run_dir / "model.glb"
        glb_obj.export(str(glb_path))

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {"directory": run_dir, "ply": ply_path, "glb": glb_path}


class ObjectPlacementTool:
    """Utility window for compositing an edited object screenshot onto the inpainted background."""

    def __init__(self, workflow: "WorkflowGUI") -> None:
        self.workflow = workflow
        self.window_tag = "object_tool_window"
        self.preview_texture_tag = "object_tool_preview_texture"
        self.overlay_picker_dialog_tag = "object_tool_overlay_dialog"
        self.save_dialog_tag = "object_tool_save_dialog"
        self._preview_image_tag = "object_tool_preview_image"
        self._texture_registry_tag = "object_tool_texture_registry"
        self.preview_canvas_size = (512, 512)
        self.background_image: Optional[Image.Image] = None
        self.background_path: Optional[Path] = None
        self.overlay_image: Optional[Image.Image] = None
        self.overlay_path: Optional[Path] = None
        self.last_composite_full: Optional[Image.Image] = None

        self.overlay_x: float = 0.0
        self.overlay_y: float = 0.0
        self.overlay_scale: float = 1.0
        self.overlay_rotation: float = 0.0
        self.overlay_opacity: float = 1.0

        self._x_input_tag = "object_tool_x_input"
        self._y_input_tag = "object_tool_y_input"
        self._scale_slider_tag = "object_tool_scale_slider"
        self._rotation_slider_tag = "object_tool_rotation_slider"
        self._opacity_slider_tag = "object_tool_opacity_slider"
        self._status_text_tag = "object_tool_status_text"

        self._preview_initialized = False
        self._ensure_base_texture()

    def _ensure_base_texture(self) -> None:
        width, height = self.preview_canvas_size
        blank = np.zeros((height * width * 4,), dtype=np.float32)
        if not dpg.does_item_exist(self._texture_registry_tag):
            with dpg.texture_registry(tag=self._texture_registry_tag):
                dpg.add_dynamic_texture(width, height, blank.tolist(), tag=self.preview_texture_tag)
        elif not dpg.does_item_exist(self.preview_texture_tag):
            dpg.add_dynamic_texture(width, height, blank.tolist(), parent=self._texture_registry_tag, tag=self.preview_texture_tag)

    # ----------------------------- Public API -----------------------------
    def open(self, background_path: Path) -> None:
        try:
            self.background_image = Image.open(background_path).convert("RGBA")
        except Exception as exc:
            self.workflow._log(f"[Compose] Failed to load background: {exc}")
            return
        self.background_path = background_path
        self._ensure_window()
        self._ensure_preview()
        self._reset_overlay_position()
        dpg.configure_item(self.window_tag, show=True)
        self._set_status("请选择或加载前景截图 PNG 文件")
        self._update_preview()

    # ----------------------------- UI Construction -----------------------------
    def _ensure_window(self) -> None:
        if dpg.does_item_exist(self.window_tag):
            return

        with dpg.window(
            label="Object Placement",
            tag=self.window_tag,
            width=800,
            height=600,
            pos=(30, 30),
            show=False,
        ):
            with dpg.group(horizontal=True):
                with dpg.child_window(width=260, autosize_y=True, border=True):
                    dpg.add_text("前景控制")
                    dpg.add_separator()
                    dpg.add_button(label="加载前景 PNG", callback=self._open_overlay_picker)
                    dpg.add_spacer(height=5)
                    dpg.add_text("位置 (像素, 以背景为基准)")
                    dpg.add_input_float(label="X", tag=self._x_input_tag, default_value=0.0, step=1.0, callback=self._on_position_changed)
                    dpg.add_input_float(label="Y", tag=self._y_input_tag, default_value=0.0, step=1.0, callback=self._on_position_changed)

                    dpg.add_separator()
                    dpg.add_text("缩放")
                    dpg.add_slider_float(label="比例", tag=self._scale_slider_tag, min_value=0.1, max_value=5.0, default_value=1.0, callback=self._on_scale_changed)

                    dpg.add_separator()
                    dpg.add_text("旋转")
                    dpg.add_slider_float(label="角度", tag=self._rotation_slider_tag, min_value=-180.0, max_value=180.0, default_value=0.0, callback=self._on_rotation_changed)

                    dpg.add_separator()
                    dpg.add_text("透明度")
                    dpg.add_slider_float(label="透明度", tag=self._opacity_slider_tag, min_value=0.0, max_value=1.0, default_value=1.0, callback=self._on_opacity_changed)

                    dpg.add_separator()
                    dpg.add_button(label="重置位置", callback=self._reset_overlay_position)
                    dpg.add_spacer(height=8)
                    dpg.add_button(label="保存合成结果", callback=self._open_save_dialog)
                    dpg.add_spacer(height=8)
                    dpg.add_text("状态", color=(120, 120, 120))
                    dpg.add_text("等待前景图片", tag=self._status_text_tag, wrap=220)

                with dpg.child_window(border=True, width=-1, height=-1):
                    dpg.add_text("预览")
                    dpg.add_separator()
                    if not self._preview_initialized:
                        dpg.add_image(self.preview_texture_tag, tag=self._preview_image_tag)
                        self._preview_initialized = True

        if not dpg.does_item_exist(self.overlay_picker_dialog_tag):
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=self._on_overlay_file_selected,
                file_count=1,
                tag=self.overlay_picker_dialog_tag,
                width=700,
                height=400,
            ):
                dpg.add_file_extension("PNG 文件{.png}")

        if not dpg.does_item_exist(self.save_dialog_tag):
            with dpg.file_dialog(
                directory_selector=False,
                show=False,
                callback=self._on_save_path_selected,
                file_count=1,
                tag=self.save_dialog_tag,
                width=700,
                height=400,
                modal=True,
            ):
                dpg.add_file_extension("PNG 文件{.png}")
                dpg.add_file_extension("JPEG 文件{.jpg,.jpeg}")

    def _ensure_preview(self) -> None:
        width, height = self.preview_canvas_size
        if not dpg.does_item_exist(self.preview_texture_tag):
            self._ensure_base_texture()
        if dpg.does_item_exist(self._preview_image_tag):
            dpg.configure_item(
                self._preview_image_tag,
                texture_tag=self.preview_texture_tag,
                width=width,
                height=height,
                uv_min=(0.0, 0.0),
                uv_max=(1.0, 1.0),
            )

    # ----------------------------- Event Handlers -----------------------------
    def _open_overlay_picker(self) -> None:
        dpg.show_item(self.overlay_picker_dialog_tag)

    def _on_overlay_file_selected(self, sender, app_data) -> None:  # noqa: D401
        selections = app_data.get("selections", {})
        if not selections:
            return
        _, file_path = next(iter(selections.items()))
        try:
            overlay = Image.open(file_path).convert("RGBA")
        except Exception as exc:
            self.workflow._log(f"[Compose] 无法加载前景图片: {exc}")
            return
        self.overlay_image = overlay
        self.overlay_path = Path(file_path)
        self._reset_overlay_position()
        self._set_status(f"前景 PNG: {self.overlay_path.name}")
        self.workflow._log(f"[Compose] 前景 PNG 已加载: {self.overlay_path}")
        self._update_preview()

    def _open_save_dialog(self) -> None:
        if self.overlay_image is None:
            self._set_status("请先加载前景 PNG 并进行调节")
            return
        if self.last_composite_full is None:
            self._set_status("暂无可保存的预览，请稍后重试")
            return
        dpg.show_item(self.save_dialog_tag)

    def _on_save_path_selected(self, sender, app_data) -> None:  # noqa: D401
        selections = app_data.get("selections", {})
        if not selections:
            return
        _, file_path = next(iter(selections.items()))
        self._save_composite(Path(file_path))

    def _on_position_changed(self, sender, app_data) -> None:  # noqa: D401
        if sender == self._x_input_tag:
            self.overlay_x = float(app_data)
        elif sender == self._y_input_tag:
            self.overlay_y = float(app_data)
        self._update_preview()

    def _on_scale_changed(self, sender, app_data) -> None:  # noqa: D401
        self.overlay_scale = float(app_data)
        self._update_preview()

    def _on_rotation_changed(self, sender, app_data) -> None:  # noqa: D401
        self.overlay_rotation = float(app_data)
        self._update_preview()

    def _on_opacity_changed(self, sender, app_data) -> None:  # noqa: D401
        self.overlay_opacity = float(app_data)
        self._update_preview()

    # ----------------------------- Image Processing -----------------------------
    def _reset_overlay_position(self) -> None:
        if not self.background_image:
            return
        bg_w, bg_h = self.background_image.size
        self.overlay_x = bg_w / 2
        self.overlay_y = bg_h / 2
        self.overlay_scale = 1.0
        self.overlay_rotation = 0.0
        self.overlay_opacity = 1.0
        dpg.set_value(self._x_input_tag, self.overlay_x)
        dpg.set_value(self._y_input_tag, self.overlay_y)
        dpg.set_value(self._scale_slider_tag, self.overlay_scale)
        dpg.set_value(self._rotation_slider_tag, self.overlay_rotation)
        dpg.set_value(self._opacity_slider_tag, self.overlay_opacity)

    def _generate_composite(self) -> Optional[Image.Image]:
        if self.background_image is None:
            return None
        result = self.background_image.copy()
        if self.overlay_image is None:
            return result

        overlay = self.overlay_image.copy()

        if self.overlay_scale != 1.0:
            new_w = max(1, int(overlay.width * self.overlay_scale))
            new_h = max(1, int(overlay.height * self.overlay_scale))
            overlay = overlay.resize((new_w, new_h), Image.Resampling.LANCZOS)

        if self.overlay_rotation != 0.0:
            overlay = overlay.rotate(self.overlay_rotation, expand=True, resample=Image.Resampling.BICUBIC)

        if self.overlay_opacity < 1.0:
            alpha = overlay.split()[-1]
            alpha = alpha.point(lambda p: int(p * self.overlay_opacity))
            overlay.putalpha(alpha)

        paste_x = int(round(self.overlay_x - overlay.width / 2))
        paste_y = int(round(self.overlay_y - overlay.height / 2))

        result.paste(overlay, (paste_x, paste_y), overlay)
        return result

    def _update_preview(self) -> None:
        composite = self._generate_composite()
        if composite is None:
            return

        self.last_composite_full = composite
        canvas_w, canvas_h = self.preview_canvas_size
        img_w, img_h = composite.size
        scale = min(canvas_w / img_w, canvas_h / img_h, 1.0)
        target_w = max(1, int(img_w * scale))
        target_h = max(1, int(img_h * scale))
        preview_img = composite.resize((target_w, target_h), Image.Resampling.LANCZOS) if (target_w, target_h) != composite.size else composite.copy()

        canvas = Image.new("RGBA", (canvas_w, canvas_h), (240, 240, 240, 255))
        offset_x = (canvas_w - target_w) // 2
        offset_y = (canvas_h - target_h) // 2
        canvas.paste(preview_img, (offset_x, offset_y))

        data = np.asarray(canvas, dtype=np.float32) / 255.0
        dpg.set_value(self.preview_texture_tag, data.flatten().tolist())

    # ----------------------------- Saving -----------------------------
    def _save_composite(self, output_path: Path) -> None:
        if self.last_composite_full is None:
            self._set_status("没有可保存的合成结果")
            return
        try:
            if output_path.suffix.lower() in {".jpg", ".jpeg"}:
                rgb_image = Image.new("RGB", self.last_composite_full.size, (255, 255, 255))
                rgb_image.paste(self.last_composite_full, mask=self.last_composite_full.split()[-1])
                rgb_image.save(output_path, quality=95)
            else:
                self.last_composite_full.save(output_path)
        except Exception as exc:
            self._set_status(f"保存失败: {exc}")
            self.workflow._log(f"[Compose] 保存失败: {exc}")
            return

        self._set_status(f"已保存为: {output_path.name}")
        self.workflow.on_composite_saved(output_path)

    # ----------------------------- Utilities -----------------------------
    def _set_status(self, text: str) -> None:
        if dpg.does_item_exist(self._status_text_tag):
            dpg.set_value(self._status_text_tag, text)


class WorkflowGUI:
    """Lightweight DearPyGUI interface orchestrating SAM segmentation and PixelHacker inpainting."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.sam_output_dir = Path(args.output_dir) / "sam"
        self.inpaint_output_dir = Path(args.output_dir) / "pixelhacker"
        self.trellis_output_dir = Path(args.trellis_output_dir)
        self.sam_output_dir.mkdir(parents=True, exist_ok=True)
        self.inpaint_output_dir.mkdir(parents=True, exist_ok=True)
        self.trellis_output_dir.mkdir(parents=True, exist_ok=True)

        self.sam_processor: Optional[SimplifiedSAMProcessor] = None
        self.sam_release_gpu = args.sam_release_gpu
        self.pixelhacker_release_gpu = args.pixelhacker_release_gpu
        self.pixelhacker = PixelHackerRunner(
            config_path=Path(args.pixelhacker_config),
            weight_path=Path(args.pixelhacker_weight),
            device=args.device,
            release_after_run=self.pixelhacker_release_gpu,
        )
        self.trellis_runner = TrellisRunner(
            model_id=args.trellis_model,
            device=args.trellis_device,
            simplify_ratio=args.trellis_simplify,
            texture_size=args.trellis_texture_size,
            default_seed=args.trellis_seed,
            offline_mode=args.trellis_offline,
        )
        self.trellis_release_gpu = args.trellis_release_gpu
        self.trellis_preload = args.trellis_preload
        self.trellis_ready = not self.trellis_preload
        self.scgs_script = Path(args.scgs_script)
        self.scgs_width = args.scgs_width
        self.scgs_height = args.scgs_height
        self.scgs_white_background = args.scgs_white_background

        self.source_image: Optional[Path] = None
        self.sam_results: Optional[Dict[str, str]] = None
        self.inpaint_result: Optional[Path] = None
        self.trellis_result: Optional[Dict[str, Path]] = None
        self.scgs_process: Optional[subprocess.Popen] = None
        self.composite_result: Optional[Path] = None

        self.log_messages = []
        self.pixelhacker_running = False
        self.trellis_running = False
        self._event_queue = queue.Queue()

        dpg.create_context()
        self.object_tool = ObjectPlacementTool(self)
        self._build_ui()

        if self.trellis_preload:
            self._log("[Trellis] Preloading pipeline in background...")
            threading.Thread(target=self._preload_trellis, daemon=True).start()
            dpg.configure_item("run_trellis_button", enabled=False)

    # ----------------------------- UI Helpers -----------------------------
    def _build_ui(self) -> None:
        with dpg.window(label="2D SpaceEdit Workflow", tag="primary_window", width=900, height=620):
            dpg.add_text("Step 1 - Select Input Image")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Choose Image", callback=lambda: dpg.show_item("image_file_dialog"))
                dpg.add_text("None", tag="selected_image_text")

            dpg.add_separator()

            dpg.add_text("Step 2 - Interactive SAM Segmentation")
            dpg.add_button(label="Open SAM Selector", tag="run_sam_button", callback=self._handle_run_sam)
            dpg.add_text("Mask: none", tag="sam_mask_text")

            dpg.add_separator()

            dpg.add_text("Step 3 - PixelHacker Background Fill")
            dpg.add_button(label="Run PixelHacker", tag="run_pixelhacker_button", callback=self._handle_run_pixelhacker)
            dpg.add_text("Inpaint Result: none", tag="pixelhacker_output_text")

            dpg.add_separator()
            dpg.add_text("Step 4 - Trellis 3D Reconstruction")
            dpg.add_button(label="Run Trellis", tag="run_trellis_button", callback=self._handle_run_trellis)
            dpg.add_text("Trellis Output: none", tag="trellis_output_text")

            dpg.add_separator()
            dpg.add_text("Step 5 - SC-GS Editing")
            dpg.add_button(label="Open SC-GS Editor", tag="open_scgs_button", callback=self._handle_open_scgs)
            dpg.add_text("SC-GS Status: idle", tag="scgs_status_text")

            dpg.add_separator()
            dpg.add_text("Step 6 - Object Reprojection")
            dpg.add_button(label="Open Object Placement Tool", tag="open_object_tool_button", callback=self._handle_open_object_tool)
            dpg.add_text("Composite Result: none", tag="composite_result_text")

            dpg.add_separator()
            dpg.add_text("Optional - Load Existing Results")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Load Saved Mask", callback=lambda: dpg.show_item("mask_file_dialog"))
                dpg.add_button(label="Load Object PNG", callback=lambda: dpg.show_item("object_file_dialog"))
                dpg.add_button(label="Load Background Image", callback=lambda: dpg.show_item("background_file_dialog"))
                dpg.add_button(label="Load Trellis PLY", callback=lambda: dpg.show_item("ply_file_dialog"))

            dpg.add_separator()
            dpg.add_text("Console Log")
            dpg.add_input_text(tag="log_panel", multiline=True, readonly=True, width=860, height=260)

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_image_selected,
            file_count=1,
            tag="image_file_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension("Images{.jpg,.jpeg,.png,.bmp,.tiff,.webp}")

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_mask_selected,
            file_count=1,
            tag="mask_file_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension("Mask{.png,.jpg,.jpeg,.bmp}")

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_object_selected,
            file_count=1,
            tag="object_file_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension("Object PNG{.png}")

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_background_selected,
            file_count=1,
            tag="background_file_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension("Images{.jpg,.jpeg,.png,.bmp,.tiff,.webp}")

        with dpg.file_dialog(
            directory_selector=False,
            show=False,
            callback=self._on_ply_selected,
            file_count=1,
            tag="ply_file_dialog",
            width=700,
            height=400,
        ):
            dpg.add_file_extension("PLY{.ply}")

    def _log(self, message: str) -> None:
        print(message)
        self.log_messages.append(message)
        self.log_messages = self.log_messages[-200:]
        dpg.set_value("log_panel", "\n".join(self.log_messages))

    def _enqueue_event(self, event_type: str, payload: object | None = None) -> None:
        self._event_queue.put((event_type, payload))

    def _log_threadsafe(self, message: str) -> None:
        self._enqueue_event("log", message)


    def _preload_trellis(self) -> None:
        try:
            self.trellis_runner.preload()
            self._enqueue_event("trellis_preload_done", None)
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            self._enqueue_event("trellis_preload_error", str(exc))

    def _ensure_sam_processor(self) -> None:
        if self.sam_processor is not None:
            return
        checkpoint = Path(self.args.sam_checkpoint)
        if not checkpoint.exists():
            self._log(f"[SAM] Checkpoint not found: {checkpoint}")
            return
        self._log("[SAM] Loading model, please wait...")
        self.sam_processor = SimplifiedSAMProcessor(
            checkpoint_path=str(checkpoint),
            output_base_dir=str(self.sam_output_dir),
        )
        self._log("[SAM] Ready. Use the SAM window and press SPACE to save outputs.")

    def _release_sam(self) -> None:
        if self.sam_processor is None:
            return
        try:
            if hasattr(self.sam_processor, "sam"):
                self.sam_processor.sam.to("cpu")
            if hasattr(self.sam_processor, "predictor"):
                self.sam_processor.predictor = None
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            self._log(f"[SAM] Release warning: {exc}")
        self.sam_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._log("[SAM] GPU resources released.")

    # ----------------------------- Callbacks -----------------------------
    def _on_image_selected(self, sender, app_data) -> None:  # noqa: D401 - DearPyGUI signature
        for display_name, file_path in app_data["selections"].items():
            self.source_image = Path(file_path)
            dpg.set_value("selected_image_text", display_name)
            self._log(f"[Input] Image selected: {file_path}")
            self.sam_results = None
            self.inpaint_result = None
            self.trellis_result = None
            self.composite_result = None
            dpg.set_value("sam_mask_text", "Mask: none")
            dpg.set_value("pixelhacker_output_text", "Inpaint Result: none")
            dpg.set_value("trellis_output_text", "Trellis Output: none")
            dpg.set_value("composite_result_text", "Composite Result: none")
            break

    def _on_mask_selected(self, sender, app_data) -> None:  # noqa: D401 - DearPyGUI signature
        for _, file_path in app_data["selections"].items():
            mask_path = Path(file_path)
            if not mask_path.exists():
                self._log(f"[Mask] File not found: {file_path}")
                break
            self.sam_results = self.sam_results or {}
            self.sam_results["mask"] = str(mask_path)
            dpg.set_value("sam_mask_text", f"Mask: {mask_path}")
            self._log(f"[Mask] Using existing mask: {file_path}")
            break

    def _on_object_selected(self, sender, app_data) -> None:  # noqa: D401 - DearPyGUI signature
        for _, file_path in app_data["selections"].items():
            object_path = Path(file_path)
            if not object_path.exists():
                self._log(f"[Trellis] File not found: {file_path}")
                break
            self.sam_results = self.sam_results or {}
            self.sam_results["object"] = str(object_path)
            dpg.set_value("trellis_output_text", f"Object PNG: {object_path}")
            self._log(f"[Trellis] Using existing object PNG: {file_path}")
            break

    def _on_background_selected(self, sender, app_data) -> None:  # noqa: D401 - DearPyGUI signature
        for _, file_path in app_data["selections"].items():
            background_path = Path(file_path)
            if not background_path.exists():
                self._log(f"[Compose] File not found: {file_path}")
                break
            try:
                with Image.open(background_path) as _:
                    pass
            except Exception as exc:
                self._log(f"[Compose] Failed to load background: {exc}")
                break
            self.inpaint_result = background_path
            self.composite_result = None
            dpg.set_value("pixelhacker_output_text", f"Inpaint Result: {background_path}")
            dpg.set_value("composite_result_text", "Composite Result: none")
            self._log(f"[Compose] Using existing background image: {file_path}")
            break

    def _on_ply_selected(self, sender, app_data) -> None:  # noqa: D401 - DearPyGUI signature
        for _, file_path in app_data["selections"].items():
            ply_path = Path(file_path)
            if not ply_path.exists():
                self._log(f"[Trellis] File not found: {file_path}")
                break
            self.trellis_result = {
                "directory": ply_path.parent,
                "ply": ply_path,
                "glb": None,
            }
            dpg.set_value("trellis_output_text", f"Trellis Output: {ply_path}")
            self._log(f"[Trellis] Using existing PLY: {file_path}")
            break

    def _handle_run_sam(self) -> None:
        if self.source_image is None:
            self._log("[SAM] Please select an input image first.")
            return
        self._ensure_sam_processor()
        if self.sam_processor is None:
            self._log("[SAM] Model not available, cannot start.")
            return

        self._log("[SAM] Opening interactive window. Follow the on-screen instructions and press SPACE to save.")
        result = self.sam_processor.process_image(str(self.source_image), auto_exit_on_save=True)
        if not result:
            self._log("[SAM] No saved results detected. Please ensure SPACE/ENTER was pressed in the SAM window.")
            return

        self.sam_results = result
        mask_path = result.get("mask")
        if mask_path:
            dpg.set_value("sam_mask_text", f"Mask: {mask_path}")
        self._log(f"[SAM] Done. Outputs stored under {self.sam_output_dir}")
        if self.sam_release_gpu:
            self._release_sam()

    def _handle_run_pixelhacker(self) -> None:
        if self.source_image is None:
            self._log("[PixelHacker] Please select an input image first.")
            return
        if not self.sam_results or not self.sam_results.get("mask"):
            self._log("[PixelHacker] Please run SAM and save a mask before inpainting.")
            return

        if self.pixelhacker_running:
            self._log("[PixelHacker] A job is already running. Please wait...")
            return

        mask_path = Path(self.sam_results["mask"])
        if not mask_path.exists():
            self._log(f"[PixelHacker] Mask file not found: {mask_path}")
            return

        output_path = self.inpaint_output_dir / f"{self.source_image.stem}_inpaint.png"
        self.pixelhacker_running = True
        dpg.configure_item("run_pixelhacker_button", enabled=False)
        self._log("[PixelHacker] Running background inpainting. This may take a while...")

        thread = threading.Thread(
            target=self._pixelhacker_worker,
            args=(self.source_image, mask_path, output_path),
            daemon=True,
        )
        thread.start()

    def _handle_run_trellis(self) -> None:
        if not self.trellis_ready:
            self._log("[Trellis] Still loading. Please wait for preload to finish.")
            return
        if not self.sam_results or not self.sam_results.get("object"):
            self._log("[Trellis] Please run SAM and save the object PNG first.")
            return

        object_path = Path(self.sam_results["object"])
        if not object_path.exists():
            self._log(f"[Trellis] Object file not found: {object_path}")
            return

        if self.trellis_running:
            self._log("[Trellis] A generation job is already running. Please wait...")
            return

        if self.pixelhacker_release_gpu:
            self.pixelhacker.release()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.trellis_running = True
        dpg.configure_item("run_trellis_button", enabled=False)
        self._log("[Trellis] Generating 3D Gaussian splats, this may take a while...")

        thread = threading.Thread(
            target=self._trellis_worker,
            args=(object_path,),
            daemon=True,
        )
        thread.start()

    def _handle_open_scgs(self) -> None:
        if self.scgs_process and self.scgs_process.poll() is None:
            self._log("[SC-GS] Editor already running. Close it before launching again.")
            return
        if not self.trellis_result or not self.trellis_result.get("ply"):
            self._log("[SC-GS] Please run Trellis to generate a PLY file first.")
            return

        ply_path = self.trellis_result.get("ply")
        if ply_path is None or not Path(ply_path).exists():
            self._log(f"[SC-GS] PLY file not found: {ply_path}")
            return

        # Release Trellis resources before launching SC-GS to ensure free GPU memory
        if self.trellis_runner.pipeline is not None:
            self._log("[SC-GS] Releasing Trellis pipeline before launch.")
            self.trellis_runner.release()
            if self.trellis_preload:
                self.trellis_ready = False
                dpg.configure_item("run_trellis_button", enabled=False)
                threading.Thread(target=self._preload_trellis, daemon=True).start()

        command = [
            sys.executable,
            str(self.scgs_script),
            "--gs_path",
            str(ply_path),
            "--gui",
            "--W",
            str(self.scgs_width),
            "--H",
            str(self.scgs_height),
        ]
        if self.scgs_white_background:
            command.append("--white_background")

        self._log("[SC-GS] Launching editor...")
        dpg.set_value("scgs_status_text", "SC-GS Status: launching...")
        try:
            process = subprocess.Popen(command, cwd=str(REPO_ROOT))
        except Exception as exc:
            self._log(f"[SC-GS] Failed to launch: {exc}")
            dpg.set_value("scgs_status_text", "SC-GS Status: failed")
            return

        self.scgs_process = process
        dpg.set_value("scgs_status_text", "SC-GS Status: running")
        self._log("[SC-GS] Editor started. Close the SC-GS window to return.")

        watcher = threading.Thread(target=self._wait_for_scgs, args=(process,), daemon=True)
        watcher.start()

    def _handle_open_object_tool(self) -> None:
        if self.inpaint_result is None:
            self._log("[Compose] 请先完成 PixelHacker 修复以生成背景图像。")
            return
        if not self.inpaint_result.exists():
            self._log(f"[Compose] 找不到背景图片: {self.inpaint_result}")
            return
        self._log(f"[Compose] 打开合成工具，背景: {self.inpaint_result}")
        self.object_tool.open(self.inpaint_result)

    def _wait_for_scgs(self, process: subprocess.Popen) -> None:
        try:
            return_code = process.wait()
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            self._enqueue_event("scgs_exit", -1)
            self._log_threadsafe(f"[SC-GS] Monitoring failed: {exc}")
            return
        self._enqueue_event("scgs_exit", return_code)

    def _pixelhacker_worker(self, image_path: Path, mask_path: Path, output_path: Path) -> None:
        try:
            result_path = self.pixelhacker.run(
                image_path=image_path,
                mask_path=mask_path,
                output_path=output_path,
            )
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            if self.pixelhacker_release_gpu:
                self.pixelhacker.release()
            self._log_threadsafe(f"[PixelHacker] Inpainting failed: {exc}")
            self._enqueue_event("pixelhacker_error", str(exc))
            return

        self._enqueue_event("pixelhacker_done", str(result_path))

    def _trellis_worker(self, object_path: Path) -> None:
        release_requested = self.trellis_release_gpu
        try:
            result = self.trellis_runner.run(
                image_path=object_path,
                output_root=self.trellis_output_dir,
                seed=self.args.trellis_seed,
            )
        except Exception as exc:  # pragma: no cover - runtime diagnostics
            self._log_threadsafe(f"[Trellis] Generation failed: {exc}")
            if release_requested:
                self.trellis_runner.release()
            self._enqueue_event("trellis_error", str(exc))
            if release_requested:
                self._enqueue_event("trellis_released", None)
            return

        payload = {k: str(v) for k, v in result.items()}
        self._enqueue_event("trellis_done", payload)
        if release_requested:
            self.trellis_runner.release()
            self._enqueue_event("trellis_released", None)

    def _process_event_queue(self) -> None:
        while not self._event_queue.empty():
            event_type, payload = self._event_queue.get_nowait()
            if event_type == "log" and isinstance(payload, str):
                self._log(payload)
            elif event_type == "pixelhacker_done" and isinstance(payload, str):
                self.pixelhacker_running = False
                dpg.configure_item("run_pixelhacker_button", enabled=True)
                self.inpaint_result = Path(payload)
                self.composite_result = None
                dpg.set_value("pixelhacker_output_text", f"Inpaint Result: {payload}")
                dpg.set_value("composite_result_text", "Composite Result: none")
                self._log(f"[PixelHacker] Done. Result saved to {payload}")
            elif event_type == "pixelhacker_error" and isinstance(payload, str):
                self.pixelhacker_running = False
                dpg.configure_item("run_pixelhacker_button", enabled=True)
                self._log(f"[PixelHacker] Inpainting failed: {payload}")
            elif event_type == "trellis_preload_done":
                self.trellis_ready = True
                self._log("[Trellis] Preload finished. Ready for generation.")
                dpg.configure_item("run_trellis_button", enabled=True)
            elif event_type == "trellis_preload_error" and isinstance(payload, str):
                self.trellis_ready = True
                self._log(f"[Trellis] Preload failed: {payload}")
                dpg.configure_item("run_trellis_button", enabled=True)
            elif event_type == "trellis_released":
                if self.trellis_preload:
                    self.trellis_ready = False
                    self._log("[Trellis] GPU resources released. Re-preloading...")
                    dpg.configure_item("run_trellis_button", enabled=False)
                    threading.Thread(target=self._preload_trellis, daemon=True).start()
                else:
                    self.trellis_ready = True
                    self._log("[Trellis] GPU resources released.")
                    dpg.configure_item("run_trellis_button", enabled=True)
            elif event_type == "trellis_done" and isinstance(payload, dict):
                self.trellis_running = False
                dpg.configure_item("run_trellis_button", enabled=True)
                dir_path = payload.get("directory")
                ply_path = payload.get("ply")
                glb_path = payload.get("glb")
                if dir_path:
                    dpg.set_value("trellis_output_text", f"Trellis Output: {dir_path}")
                self.trellis_result = {
                    "directory": Path(dir_path) if dir_path else None,
                    "ply": Path(ply_path) if ply_path else None,
                    "glb": Path(glb_path) if glb_path else None,
                }
                info_parts = []
                if ply_path:
                    info_parts.append(f"PLY -> {ply_path}")
                if glb_path:
                    info_parts.append(f"GLB -> {glb_path}")
                joined = "; ".join(info_parts) if info_parts else ""
                self._log(f"[Trellis] Done. {joined}")
            elif event_type == "trellis_error" and isinstance(payload, str):
                self.trellis_running = False
                dpg.configure_item("run_trellis_button", enabled=True)
                self._log(f"[Trellis] Generation failed: {payload}")
            elif event_type == "scgs_exit" and isinstance(payload, int):
                self.scgs_process = None
                status = "SC-GS Status: idle"
                dpg.set_value("scgs_status_text", status)
                self._log(f"[SC-GS] Editor closed (exit code {payload}).")

    # ----------------------------- Run Loop -----------------------------
    def run(self) -> None:
        dpg.create_viewport(title="2D-SpaceEdit Workflow", width=900, height=660, resizable=False)
        dpg.setup_dearpygui()
        dpg.show_viewport()
        while dpg.is_dearpygui_running():
            self._process_event_queue()
            dpg.render_dearpygui_frame()
        dpg.destroy_context()

    def on_composite_saved(self, output_path: Path) -> None:
        self.composite_result = output_path
        dpg.set_value("composite_result_text", f"Composite Result: {output_path}")
        self._log(f"[Compose] 合成结果已保存: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactive workflow for SAM segmentation and PixelHacker inpainting")
    parser.add_argument("--sam-checkpoint", default=str(REPO_ROOT / "checkpoints/sam/sam_vit_h_4b8939.pth"), help="Path to SAM checkpoint")
    parser.add_argument("--sam-release-gpu", action="store_true", help="Move SAM model back to CPU after each segmentation run")
    parser.add_argument("--pixelhacker-config", default=str(REPO_ROOT / "inpainting/config/PixelHacker_sdvae_f8d4.yaml"), help="PixelHacker config file")
    parser.add_argument("--pixelhacker-weight", default=str(REPO_ROOT / "inpainting/weight/ft_places2/diffusion_pytorch_model.bin"), help="PixelHacker weight file")
    parser.add_argument("--pixelhacker-release-gpu", action="store_true", help="Move PixelHacker models back to CPU after each run to free GPU memory")
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "outputs/workflow"), help="Directory to store intermediate results")
    parser.add_argument("--device", default="cuda", help="Computation device for PixelHacker (cuda or cpu)")
    parser.add_argument("--trellis-model", default="microsoft/TRELLIS-image-large", help="Trellis model identifier or local path")
    parser.add_argument("--trellis-output-dir", default=str(REPO_ROOT / "outputs/workflow/trellis"), help="Directory to store Trellis outputs")
    parser.add_argument("--trellis-device", default="cuda", help="Device for Trellis pipeline (cuda or cpu)")
    parser.add_argument("--trellis-seed", type=int, default=1, help="Random seed for Trellis generation")
    parser.add_argument("--trellis-simplify", type=float, default=0.9, help="Mesh simplification ratio for GLB export")
    parser.add_argument("--trellis-texture-size", type=int, default=1024, help="Texture resolution for GLB export")
    parser.add_argument("--trellis-offline", action="store_true", help="Enable offline mode for Trellis (HF cache only)")
    parser.add_argument("--trellis-preload", action="store_true", help="Preload Trellis pipeline at startup to reduce first-run latency")
    parser.add_argument("--trellis-release-gpu", action="store_true", help="Release Trellis pipeline memory after each generation")
    parser.add_argument("--scgs-script", default=str(REPO_ROOT / "editing/edit_gui.py"), help="Path to SC-GS edit_gui.py script")
    parser.add_argument("--scgs-width", type=int, default=800, help="SC-GS viewport width")
    parser.add_argument("--scgs-height", type=int, default=800, help="SC-GS viewport height")
    parser.add_argument("--scgs-white-background", action="store_true", help="Launch SC-GS with white background")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    gui = WorkflowGUI(cli_args)
    gui.run()
