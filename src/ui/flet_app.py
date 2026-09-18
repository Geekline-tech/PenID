import flet as ft
import cv2
import numpy as np
from pathlib import Path
from threading import Thread
from typing import Optional
from src.utils.config import Config, RAW_DIR, PATCHES_DIR
from src.utils.database import Database
from src.inference.gallery import Gallery
from src.data.preprocess import prepare_person_data
from src.data.scanner import scan_document


def safe_imread(path: str, flags=cv2.IMREAD_GRAYSCALE):
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, flags)


def border_all(width: int, color: str):
    side = ft.BorderSide(width, color)
    return ft.Border(top=side, bottom=side, left=side, right=side)


class PenIDApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Pen ID - 手写笔迹识别系统"
        self.page.theme_mode = ft.ThemeMode.DARK
        self.page.theme = ft.Theme(
            color_scheme_seed="#89b4fa",
            use_material3=True,
        )
        self.page.window.width = 1100
        self.page.window.height = 750
        self.page.padding = 0

        self.gallery = Gallery()
        try:
            self.gallery.load_model()
            self.model_loaded = True
        except Exception:
            self.model_loaded = False

        self.current_image: Optional[np.ndarray] = None

        self.build_ui()
        self.page.update()

    def build_ui(self):
        self.nav_rail = ft.NavigationRail(
            selected_index=0,
            label_type=ft.NavigationRailLabelType.ALL,
            min_width=100,
            min_extended_width=200,
            destinations=[
                ft.NavigationRailDestination(icon=ft.Icons.SEARCH, label="识别"),
                ft.NavigationRailDestination(icon=ft.Icons.MODEL_TRAINING, label="训练"),
                ft.NavigationRailDestination(icon=ft.Icons.PEOPLE, label="人员"),
            ],
            on_change=self.on_nav_change,
        )

        self.content_area = ft.Container(expand=True)
        self.show_identify_page()

        self.page.add(
            ft.Row(
                [
                    self.nav_rail,
                    ft.VerticalDivider(width=1),
                    self.content_area,
                ],
                expand=True,
                spacing=0,
            )
        )

    def on_nav_change(self, e):
        idx = e.control.selected_index
        if idx == 0:
            self.show_identify_page()
        elif idx == 1:
            self.show_train_page()
        else:
            self.show_gallery_page()

    def clear_content(self):
        self.content_area.content = None

    def show_identify_page(self):
        self.clear_content()

        self.upload_image_view = ft.Container(
            content=ft.Column(
                [
                    ft.Icon(ft.Icons.UPLOAD_FILE, size=48, color="#6c7086"),
                    ft.Text("点击上传笔迹图片", color="#6c7086", size=14),
                ],
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                alignment=ft.MainAxisAlignment.CENTER,
            ),
            width=500,
            height=250,
            border=border_all(2, "#45475a"),
            border_radius=12,
            bgcolor="#11111b",
            alignment=ft.Alignment(0, 0),
        )

        self.result_name = ft.Text("--", size=24, weight=ft.FontWeight.BOLD, color="#89b4fa")
        self.result_conf = ft.Text("置信度: --", size=14, color="#a6adc8")
        self.result_status = ft.Text("", size=13, weight=ft.FontWeight.BOLD)
        self.detail_text = ft.Text("", size=12, color="#a6adc8")
        self.rankings_column = ft.Column(spacing=0)

        self.identify_btn = ft.ElevatedButton(
            "开始识别",
            icon=ft.Icons.SEARCH,
            disabled=True,
            on_click=self.do_identify,
            style=ft.ButtonStyle(bgcolor="#89b4fa", color="#1e1e2e"),
        )

        upload_btn = ft.ElevatedButton(
            "上传图片",
            icon=ft.Icons.UPLOAD,
            on_click=self.upload_image,
        )

        scan_btn = ft.ElevatedButton(
            "扫描增强",
            icon=ft.Icons.CAMERA_ALT,
            on_click=self.scan_image,
        )

        clear_btn = ft.OutlinedButton("清除", on_click=self.clear_identify)

        result_card = ft.Container(
            content=ft.Column(
                [
                    ft.Row([self.result_name, self.result_status]),
                    self.result_conf,
                    self.detail_text,
                    ft.Divider(height=1, color="#313244"),
                    self.rankings_column,
                ],
                spacing=8,
            ),
            padding=20,
            border=border_all(2, "#45475a"),
            border_radius=12,
            bgcolor="#181825",
        )

        self.content_area.content = ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text("上传笔迹图片", size=16, weight=ft.FontWeight.BOLD),
                                self.upload_image_view,
                                ft.Row([upload_btn, scan_btn, self.identify_btn, clear_btn]),
                            ],
                            spacing=12,
                        ),
                        padding=24,
                    ),
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text("识别结果", size=16, weight=ft.FontWeight.BOLD),
                                result_card,
                            ],
                            spacing=12,
                        ),
                        padding=ft.Padding(left=24, right=24, bottom=24, top=0),
                    ),
                ],
                spacing=0,
                scroll=ft.ScrollMode.AUTO,
            ),
            expand=True,
        )
        self.page.update()

    async def upload_image(self, e):
        picker = ft.FilePicker()
        self.page.services.append(picker)
        self.page.update()
        files = await picker.pick_files(allowed_extensions=["png", "jpg", "jpeg", "bmp"])
        if files:
            path = files[0].path
            img = safe_imread(path)
            if img is not None:
                self.current_image = img
                self.show_image_preview(img)
                self.identify_btn.disabled = False
                self.page.update()

    async def scan_image(self, e):
        picker = ft.FilePicker()
        self.page.services.append(picker)
        self.page.update()
        files = await picker.pick_files(allowed_extensions=["png", "jpg", "jpeg", "bmp"])
        if files:
            path = files[0].path
            raw = safe_imread(path, cv2.IMREAD_COLOR)
            if raw is not None:
                def run():
                    try:
                        scanned = scan_document(raw)
                        self.current_image = scanned
                        self.show_scanned_preview(raw, scanned)
                        self.identify_btn.disabled = False
                    except Exception as ex:
                        self.result_status.value = f"扫描失败: {ex}"
                        self.result_status.color = "#f38ba8"
                    self.page.update()
                Thread(target=run, daemon=True).start()

    def show_scanned_preview(self, raw: np.ndarray, scanned: np.ndarray):
        import base64
        _, buf1 = cv2.imencode(".png", raw)
        b64_raw = base64.b64encode(buf1).decode("utf-8")
        _, buf2 = cv2.imencode(".png", scanned)
        b64_scan = base64.b64encode(buf2).decode("utf-8")
        self.upload_image_view.content = ft.Row(
            [
                ft.Column(
                    [
                        ft.Text("原图", size=12, color="#6c7086"),
                        ft.Image(
                            src=f"data:image/png;base64,{b64_raw}",
                            width=230,
                            height=220,
                            fit=ft.BoxFit.CONTAIN,
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=4,
                ),
                ft.Icon(ft.Icons.ARROW_FORWARD, color="#89b4fa"),
                ft.Column(
                    [
                        ft.Text("扫描结果", size=12, color="#a6e3a1"),
                        ft.Image(
                            src=f"data:image/png;base64,{b64_scan}",
                            width=230,
                            height=220,
                            fit=ft.BoxFit.CONTAIN,
                        ),
                    ],
                    horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=4,
                ),
            ],
            alignment=ft.MainAxisAlignment.CENTER,
            vertical_alignment=ft.CrossAxisAlignment.CENTER,
        )
        self.page.update()

    def show_image_preview(self, img: np.ndarray):
        import base64
        _, buf = cv2.imencode(".png", img)
        b64 = base64.b64encode(buf).decode("utf-8")
        self.upload_image_view.content = ft.Image(
            src=f"data:image/png;base64,{b64}",
            width=480,
            height=240,
            fit=ft.BoxFit.CONTAIN,
        )
        self.page.update()

    def do_identify(self, e):
        if self.current_image is None:
            return
        self.identify_btn.disabled = True
        self.rankings_column.controls.clear()
        self.page.update()

        def run():
            try:
                result = self.gallery.identify_image(self.current_image)
                self.result_name.value = result["name"]
                self.result_conf.value = f"置信度: {result['confidence'] * 100:.1f}%"
                if result["matched"]:
                    self.result_status.value = "匹配成功"
                    self.result_status.color = "#a6e3a1"
                else:
                    self.result_status.value = "无法匹配"
                    self.result_status.color = "#f38ba8"
                self.detail_text.value = f"余弦距离: {result['distance']:.4f} | Top {len(result['rankings'])} 排行"

                for i, r in enumerate(result["rankings"]):
                    rank = i + 1
                    bar_width = int(r["confidence"] * 300)
                    color = "#a6e3a1" if r["distance"] < 0.6 else "#89b4fa" if r["distance"] < 0.8 else "#6c7086"
                    row = ft.Container(
                        content=ft.Column(
                            [
                                ft.Row(
                                    [
                                        ft.Text(f"#{rank}", size=12, weight=ft.FontWeight.BOLD, color=color, width=30),
                                        ft.Text(r["name"], size=13, weight=ft.FontWeight.BOLD, color="#cdd6f4", expand=True),
                                        ft.Text(f"{r['confidence']*100:.1f}%", size=12, color=color, width=50),
                                        ft.Text(f"d={r['distance']:.3f}", size=11, color="#6c7086", width=60),
                                    ],
                                    alignment=ft.MainAxisAlignment.START,
                                ),
                                ft.Container(
                                    content=ft.Container(
                                        bgcolor=color,
                                        border_radius=4,
                                        width=bar_width,
                                        height=6,
                                    ),
                                    bgcolor="#313244",
                                    border_radius=4,
                                    width=300,
                                    height=6,
                                ),
                            ],
                            spacing=4,
                        ),
                        padding=ft.Padding(left=0, right=0, top=4, bottom=4),
                    )
                    self.rankings_column.controls.append(row)

                self.identify_btn.disabled = False
            except Exception as ex:
                self.result_status.value = f"识别失败: {ex}"
                self.result_status.color = "#f38ba8"
                self.identify_btn.disabled = False
            self.page.update()

        Thread(target=run, daemon=True).start()

    def clear_identify(self, e):
        self.current_image = None
        self.upload_image_view.content = ft.Column(
            [
                ft.Icon(ft.Icons.UPLOAD_FILE, size=48, color="#6c7086"),
                ft.Text("点击上传笔迹图片", color="#6c7086", size=14),
            ],
            horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            alignment=ft.MainAxisAlignment.CENTER,
        )
        self.result_name.value = "--"
        self.result_conf.value = "置信度: --"
        self.result_status.value = ""
        self.detail_text.value = ""
        self.rankings_column.controls.clear()
        self.identify_btn.disabled = True
        self.page.update()

    def show_train_page(self):
        self.clear_content()

        db = Database()
        persons = db.get_persons()
        total_samples = sum(p["sample_count"] for p in persons)
        db.close()

        self.train_log = ft.ListView(expand=True, spacing=2)
        self.train_progress = ft.ProgressBar(width=400, color="#89b4fa", bgcolor="#313244")
        self.train_status = ft.Text("就绪", size=13, color="#a6adc8")
        self.train_btn = ft.ElevatedButton(
            "开始训练", icon=ft.Icons.PLAY_ARROW, on_click=self.start_training,
            style=ft.ButtonStyle(bgcolor="#a6e3a1", color="#1e1e2e"),
        )

        # 参数输入
        self.epochs_input = ft.TextField(value="60", label="轮次", width=100, height=50)
        self.batch_input = ft.TextField(value="32", label="Batch Size", width=100, height=50)
        self.lr_input = ft.TextField(value="0.0001", label="学习率", width=120, height=50)
        self.margin_input = ft.TextField(value="0.3", label="Triplet Margin", width=130, height=50)
        self.embed_input = ft.TextField(value="512", label="Embedding维度", width=140, height=50)

        # 参数说明卡片
        param_info = ft.Container(
            content=ft.Column(
                [
                    ft.Text("参数说明", size=14, weight=ft.FontWeight.BOLD, color="#89b4fa"),
                    ft.Text("• Triplet Margin: 正负样本对的最小距离间隔，越大区分度越强但收敛越慢", size=12, color="#a6adc8"),
                    ft.Text("• Batch Size: 每次训练的三元组数量，影响训练稳定性和显存占用", size=12, color="#a6adc8"),
                    ft.Text("• Embedding维度: 特征向量长度，512维表达力更强", size=12, color="#a6adc8"),
                    ft.Text("• 学习率: 越大收敛越快但可能震荡，建议 1e-4 ~ 1e-3", size=12, color="#a6adc8"),
                ],
                spacing=4,
            ),
            padding=16,
            border=border_all(1, "#45475a"),
            border_radius=8,
            bgcolor="#181825",
        )

        # 数据集信息
        stats_row = ft.Row(
            [
                self._stat_card("人员数", str(len(persons))),
                self._stat_card("总样本", f"{total_patches:,}" if (total_patches := total_samples) else "0"),
                self._stat_card("每人均值", f"{total_samples // max(len(persons), 1):,}"),
                self._stat_card("类别数", str(len(persons))),
            ],
            spacing=12,
        )

        self.content_area.content = ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text("模型训练", size=20, weight=ft.FontWeight.BOLD),
                                ft.Text("配置训练参数并启动训练", size=13, color="#a6adc8"),
                                ft.Divider(height=1, color="#313244"),
                                stats_row,
                                ft.Text("训练参数", size=14, weight=ft.FontWeight.BOLD),
                                ft.Row(
                                    [self.epochs_input, self.batch_input, self.lr_input, self.margin_input, self.embed_input],
                                    spacing=12,
                                ),
                                param_info,
                                ft.Row([self.train_btn]),
                                ft.Row([self.train_progress, self.train_status]),
                            ],
                            spacing=12,
                        ),
                        padding=24,
                    ),
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Text("训练日志", size=14, weight=ft.FontWeight.BOLD),
                                self.train_log,
                            ],
                            spacing=8,
                        ),
                        padding=ft.Padding(left=24, right=24, bottom=24, top=0),
                        expand=True,
                    ),
                ],
                spacing=0,
                scroll=ft.ScrollMode.AUTO,
            ),
            expand=True,
        )
        self.page.update()

    def _stat_card(self, label: str, value: str):
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(value, size=22, weight=ft.FontWeight.BOLD, color="#89b4fa"),
                    ft.Text(label, size=12, color="#a6adc8"),
                ],
                spacing=2,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            padding=16,
            border=border_all(1, "#45475a"),
            border_radius=8,
            bgcolor="#181825",
            width=150,
            alignment=ft.Alignment(0, 0),
        )

    def start_training(self, e):
        self.train_btn.disabled = True
        self.page.update()

        def run():
            try:
                config = Config()
                config.update({
                    "epochs": int(self.epochs_input.value),
                    "batch_size": int(self.batch_input.value),
                    "lr": float(self.lr_input.value),
                    "triplet_margin": float(self.margin_input.value),
                    "embed_dim": int(self.embed_input.value),
                })

                self.train_log.controls.append(ft.Text("开始训练...", color="#a6e3a1"))
                self.page.update()

                from src.training.trainer import Trainer

                def log_cb(msg):
                    self.train_log.controls.append(ft.Text(msg, size=11, color="#cdd6f4"))
                    self.page.update()

                trainer = Trainer(config, progress_callback=log_cb)
                trainer.train()

                self.train_log.controls.append(ft.Text("正在重建特征库...", color="#a6adc8"))
                self.page.update()

                self.gallery.load_model()
                self.gallery.build_gallery()
                self.gallery._gallery_cache = None

                self.train_log.controls.append(ft.Text(f"训练完成! 最佳准确率: {trainer.best_acc:.1f}%", color="#a6e3a1", weight=ft.FontWeight.BOLD))
                self.train_status.value = f"完成 - 最佳准确率: {trainer.best_acc:.1f}%"
            except Exception as ex:
                self.train_log.controls.append(ft.Text(f"训练失败: {ex}", color="#f38ba8"))
                self.train_status.value = "训练失败"
            finally:
                self.train_btn.disabled = False
                self.page.update()

        Thread(target=run, daemon=True).start()

    def show_gallery_page(self):
        self.clear_content()

        db = Database()
        persons = db.get_persons()
        db.close()

        self.gallery_table = self._build_person_table(persons)

        self.content_area.content = ft.Container(
            content=ft.Column(
                [
                    ft.Container(
                        content=ft.Column(
                            [
                                ft.Row(
                                    [
                                        ft.Text("人员管理", size=20, weight=ft.FontWeight.BOLD),
                                        ft.Container(expand=True),
                                        ft.ElevatedButton(
                                            "添加人员", icon=ft.Icons.ADD,
                                            on_click=self.add_person,
                                            style=ft.ButtonStyle(bgcolor="#89b4fa", color="#1e1e2e"),
                                        ),
                                        ft.ElevatedButton(
                                            "导入图片", icon=ft.Icons.UPLOAD,
                                            on_click=self.import_images,
                                        ),
                                        ft.OutlinedButton(
                                            "刷新", icon=ft.Icons.REFRESH,
                                            on_click=self.refresh_gallery,
                                        ),
                                    ],
                                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                ),
                                self.gallery_table,
                            ],
                            spacing=12,
                        ),
                        padding=24,
                    ),
                ],
                spacing=0,
            ),
            expand=True,
        )
        self.page.update()

    def _build_person_table(self, persons: list[dict]):
        rows = []
        for p in persons:
            status = "已生成" if p["has_embedding"] else "未生成"
            status_color = "#a6e3a1" if p["has_embedding"] else "#f38ba8"
            rows.append(
                ft.DataRow(
                    cells=[
                        ft.DataCell(ft.Text(str(p["id"]), size=12)),
                        ft.DataCell(ft.Text(p["name"], size=12, weight=ft.FontWeight.BOLD)),
                        ft.DataCell(ft.Text(str(p["sample_count"]), size=12)),
                        ft.DataCell(ft.Text(status, size=12, color=status_color)),
                    ]
                )
            )

        return ft.DataTable(
            columns=[
                ft.DataColumn(ft.Text("ID", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("姓名", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("样本数", weight=ft.FontWeight.BOLD)),
                ft.DataColumn(ft.Text("特征向量", weight=ft.FontWeight.BOLD)),
            ],
            rows=rows,
            border=border_all(1, "#45475a"),
            border_radius=8,
            bgcolor="#181825",
            column_spacing=40,
            horizontal_lines=ft.BorderSide(1, "#313244"),
        )

    def add_person(self, e):
        def on_submit(dialog, name_field):
            name = name_field.value.strip()
            if name:
                db = Database()
                db.add_person(name)
                db.close()
                dialog.open = False
                self.refresh_gallery(e)
                self.page.update()

        name_field = ft.TextField(label="人员姓名", autofocus=True)
        dialog = ft.AlertDialog(
            title=ft.Text("添加人员"),
            content=name_field,
            actions=[
                ft.TextButton("取消", on_click=lambda _: setattr(dialog, 'open', False) or self.page.update()),
                ft.TextButton("确定", on_click=lambda _: on_submit(dialog, name_field)),
            ],
        )
        self.page.overlay.append(dialog)
        dialog.open = True
        self.page.update()

    async def import_images(self, e):
        picker = ft.FilePicker()
        self.page.services.append(picker)
        self.page.update()
        files = await picker.pick_files(allowed_extensions=["jpg", "jpeg", "png", "bmp"])
        if files:
            dir_path = Path(files[0].path).parent
            self._do_import(dir_path)

    def _do_import(self, raw_dir: Path):
        person_name = raw_dir.name
        db = Database()
        person = db.get_person_by_name(person_name)
        if not person:
            person_id = db.add_person(person_name)
        else:
            person_id = person["id"]
        db.close()

        def run():
            try:
                self.gallery.import_person(person_name, raw_dir)
                snack = ft.SnackBar(ft.Text(f"已导入 {person_name}"), open=True)
                self.page.overlay.append(snack)
                self.page.update()
                self.refresh_gallery(None)
            except Exception as ex:
                snack = ft.SnackBar(ft.Text(f"导入失败: {ex}"), open=True)
                self.page.overlay.append(snack)
                self.page.update()

        Thread(target=run, daemon=True).start()

    def refresh_gallery(self, e):
        db = Database()
        persons = db.get_persons()
        db.close()
        self.gallery_table = self._build_person_table(persons)
        self.show_gallery_page()


def main():
    def app(page: ft.Page):
        PenIDApp(page)

    ft.app(target=app)


if __name__ == "__main__":
    main()
