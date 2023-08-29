# coding=utf-8
import time
import os
import gradio as gr
import utils
import argparse
import logging

import torch
from torch import no_grad, LongTensor
import gradio.processing_utils as gr_processing_utils
from colorama import Back, Style

import utils.commons as commons
from utils.models import SynthesizerTrn
from text import text_to_sequence

logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('asyncio').setLevel(logging.WARNING)
logging.getLogger('charset_normalizer').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('jieba').setLevel(logging.WARNING)

# limit text and audio length in huggingface spaces
limitation = os.getenv("SYSTEM") == "spaces"

audio_postprocess_ori = gr.Audio.postprocess

def audio_postprocess(self, y):
    data = audio_postprocess_ori(self, y)
    if data is None:
        return None
    return gr_processing_utils.encode_url_or_file_to_base64(data["name"])


def get_text(text, hps):
    text_norm, clean_text = text_to_sequence(
        text, hps.symbols, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm, clean_text


def vits(text, language, speaker_id, noise_scale, noise_scale_w, length_scale):
    start = time.perf_counter()
    if not len(text):
        return "输入文本不能为空！", None, None
    text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
    if len(text) > 100 and limitation:
        return f"输入文字过长！{len(text)}>100", None, None
    if language == 0:
        text = f"[ZH]{text}[ZH]"
    elif language == 1:
        text = f"[JA]{text}[JA]"
    else:
        text = f"{text}"
    stn_tst, clean_text = get_text(text, hps_ms)

    with no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
        speaker_id = LongTensor([speaker_id]).to(device)

        audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=speaker_id, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                               length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()


    return "生成成功!", (22050, audio), f"生成耗时 {round(time.perf_counter()-start, 2)} s"


def search_speaker(search_value):
    for s in speakers:
        if search_value == s:
            return s
    for s in speakers:
        if search_value in s:
            return s


def change_lang(language):
    if language == 0:
        return 0.6, 0.668, 1.2
    else:
        return 0.6, 0.668, 1.1


def make_gradio_app() -> gr.Blocks:
    
    with open('./public/style.css', 'r', encoding='utf-8') as fp:
        css = fp.read()

    with open('./public/download_audio.js', 'r', encoding='utf-8') as fp:
        download_audio_js = fp.read().strip()

    with gr.Blocks(theme=gr.themes.Soft(), css=css) as app:
        with gr.Tabs():
            with gr.TabItem("vits"):
                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="需要合成的文本" if limitation else "Text", lines=5, value="今天晚上吃啥好呢。", elem_id=f"input-text")
                        lang = gr.Dropdown(label="语言", choices=["中文", "日语", "中日混合（中文用[ZH][ZH]包裹起来，日文用[JA][JA]包裹起来）"],
                                           type="index", value="中文")
                        btn = gr.Button(value="合成")
                        with gr.Row():
                            search = gr.Textbox(
                                label="搜索讲述者", lines=1)
                            btn2 = gr.Button(value="搜索")
                        sid = gr.Dropdown(
                            label="讲述者", choices=speakers, type="index", value=speakers[228])
                        with gr.Row():
                            ns = gr.Slider(label="感情变化程度", minimum=0.1,
                                           maximum=1.0, step=0.1, value=0.4, interactive=True)
                            nsw = gr.Slider(label="音素发音长度", minimum=0.1,
                                            maximum=1.0, step=0.1, value=0.6, interactive=True)
                            ls = gr.Slider(label="控制整体语速", minimum=0.1,
                                           maximum=2.0, step=0.1, value=1.2, interactive=True)
                    with gr.Column():
                        o1 = gr.Textbox(label="输出信息")
                        o2 = gr.Audio(label="输出音频",
                                      elem_id=f"tts-audio")
                        o3 = gr.Textbox(label="额外的信息")
                        download = gr.Button("下载音频")
                    
                    btn.click(vits, inputs=[input_text, lang, sid, ns, nsw, ls], outputs=[o1, o2, o3])
                    download.click(None, [], [], _js=download_audio_js)
                    btn2.click(search_speaker, inputs=[search], outputs=[sid])
                    lang.change(change_lang, inputs=[
                                lang], outputs=[ns, nsw, ls])
            with gr.TabItem("可用人物一览"):
                gr.Radio(label="角色", choices=speakers,
                         interactive=False, type="index")
    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--checkpoint', type=str, default='./model/G_953000.pth')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--api', action="store_true", default=False)
    parser.add_argument("--share", action="store_true",
                        default=False, help="share gradio app")

    args = parser.parse_args()
    device = torch.device(args.device)
    port = args.port
    checkpoint_path = args.checkpoint

    hps_ms = utils.get_hparams_from_file(r'./model/config.json')
    net_g_ms = SynthesizerTrn(
        len(hps_ms.symbols),
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model)
    _ = net_g_ms.eval().to(device)

    speakers = hps_ms.speakers
    model, optimizer, learning_rate, epochs = utils.load_checkpoint(checkpoint_path, net_g_ms, None)
    app = make_gradio_app()

    app.queue(concurrency_count=1, api_open=args.api)
    app.launch(share=args.share, server_port=port)
