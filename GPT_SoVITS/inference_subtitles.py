'''
按中英混合识别
按日英混合识别
多语种启动切分识别语种
全部按中文识别
全部按英文识别
全部按日文识别
'''
import os, re, logging
import LangSegment
logging.getLogger("markdown_it").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpcore").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)
logging.getLogger("charset_normalizer").setLevel(logging.ERROR)
logging.getLogger("torchaudio._extension").setLevel(logging.ERROR)
import pdb
import torch
import shutil

if os.path.exists("./gweight.txt"):
    with open("./gweight.txt", 'r', encoding="utf-8") as file:
        gweight_data = file.read()
        gpt_path = os.environ.get(
            "gpt_path", gweight_data)
else:
    gpt_path = os.environ.get(
        "gpt_path", "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt")

if os.path.exists("./sweight.txt"):
    with open("./sweight.txt", 'r', encoding="utf-8") as file:
        sweight_data = file.read()
        sovits_path = os.environ.get("sovits_path", sweight_data)
else:
    sovits_path = os.environ.get("sovits_path", "GPT_SoVITS/pretrained_models/s2G488k.pth")
# gpt_path = os.environ.get(
#     "gpt_path", "pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
# )
# sovits_path = os.environ.get("sovits_path", "pretrained_models/s2G488k.pth")
cnhubert_base_path = os.environ.get(
    "cnhubert_base_path", "GPT_SoVITS/pretrained_models/chinese-hubert-base"
)
bert_path = os.environ.get(
    "bert_path", "GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large"
)
infer_ttswebui = os.environ.get("infer_ttswebui", 9872)
infer_ttswebui = int(infer_ttswebui)
is_share = os.environ.get("is_share", "False")
is_share = eval(is_share)
if "_CUDA_VISIBLE_DEVICES" in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["_CUDA_VISIBLE_DEVICES"]
is_half = eval(os.environ.get("is_half", "True")) and torch.cuda.is_available()
import gradio as gr
from transformers import AutoModelForMaskedLM, AutoTokenizer
import numpy as np
import librosa
from feature_extractor import cnhubert

cnhubert.cnhubert_base_path = cnhubert_base_path

from module.models import SynthesizerTrn
from AR.models.t2s_lightning_module import Text2SemanticLightningModule
from text import cleaned_text_to_sequence
from text.cleaner import clean_text
from time import time as ttime
from module.mel_processing import spectrogram_torch
from my_utils import load_audio
from tools.i18n.i18n import I18nAuto

i18n = I18nAuto()

# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 确保直接启动推理UI时也能够设置。

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

tokenizer = AutoTokenizer.from_pretrained(bert_path)
bert_model = AutoModelForMaskedLM.from_pretrained(bert_path)
if is_half == True:
    bert_model = bert_model.half().to(device)
else:
    bert_model = bert_model.to(device)

# denoising

import ffmpeg
import urllib.request
urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/UVR-HP2.pth", "uvr5/uvr5_weights/UVR-HP2.pth")
urllib.request.urlretrieve("https://download.openxlab.org.cn/models/Kevin676/rvc-models/weight/UVR-HP5.pth", "uvr5/uvr5_weights/UVR-HP5.pth")

from uvr5.vr import AudioPre
weight_uvr5_root = "uvr5/uvr5_weights"
uvr5_names = []
for name in os.listdir(weight_uvr5_root):
    if name.endswith(".pth") or "onnx" in name:
        uvr5_names.append(name.replace(".pth", ""))

func = AudioPre

pre_fun_hp2 = func(
  agg=int(10),
  model_path=os.path.join(weight_uvr5_root, "UVR-HP2.pth"),
  device="cuda",
  is_half=True,
)
pre_fun_hp5 = func(
  agg=int(10),
  model_path=os.path.join(weight_uvr5_root, "UVR-HP5.pth"),
  device="cuda",
  is_half=True,
)


def get_bert_feature(text, word2ph):
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt")
        for i in inputs:
            inputs[i] = inputs[i].to(device)
        res = bert_model(**inputs, output_hidden_states=True)
        res = torch.cat(res["hidden_states"][-3:-2], -1)[0].cpu()[1:-1]
    assert len(word2ph) == len(text)
    phone_level_feature = []
    for i in range(len(word2ph)):
        repeat_feature = res[i].repeat(word2ph[i], 1)
        phone_level_feature.append(repeat_feature)
    phone_level_feature = torch.cat(phone_level_feature, dim=0)
    return phone_level_feature.T


class DictToAttrRecursive(dict):
    def __init__(self, input_dict):
        super().__init__(input_dict)
        for key, value in input_dict.items():
            if isinstance(value, dict):
                value = DictToAttrRecursive(value)
            self[key] = value
            setattr(self, key, value)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")

    def __setattr__(self, key, value):
        if isinstance(value, dict):
            value = DictToAttrRecursive(value)
        super(DictToAttrRecursive, self).__setitem__(key, value)
        super().__setattr__(key, value)

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError:
            raise AttributeError(f"Attribute {item} not found")


ssl_model = cnhubert.get_model()
if is_half == True:
    ssl_model = ssl_model.half().to(device)
else:
    ssl_model = ssl_model.to(device)


def change_sovits_weights(sovits_path):
    global vq_model, hps
    dict_s2 = torch.load(sovits_path, map_location="cpu")
    hps = dict_s2["config"]
    hps = DictToAttrRecursive(hps)
    hps.model.semantic_frame_rate = "25hz"
    vq_model = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model
    )
    if ("pretrained" not in sovits_path):
        del vq_model.enc_q
    if is_half == True:
        vq_model = vq_model.half().to(device)
    else:
        vq_model = vq_model.to(device)
    vq_model.eval()
    print(vq_model.load_state_dict(dict_s2["weight"], strict=False))
    with open("./sweight.txt", "w", encoding="utf-8") as f:
        f.write(sovits_path)


change_sovits_weights(sovits_path)


def change_gpt_weights(gpt_path):
    global hz, max_sec, t2s_model, config
    hz = 50
    dict_s1 = torch.load(gpt_path, map_location="cpu")
    config = dict_s1["config"]
    max_sec = config["data"]["max_sec"]
    t2s_model = Text2SemanticLightningModule(config, "****", is_train=False)
    t2s_model.load_state_dict(dict_s1["weight"])
    if is_half == True:
        t2s_model = t2s_model.half()
    t2s_model = t2s_model.to(device)
    t2s_model.eval()
    total = sum([param.nelement() for param in t2s_model.parameters()])
    print("Number of parameter: %.2fM" % (total / 1e6))
    with open("./gweight.txt", "w", encoding="utf-8") as f: f.write(gpt_path)


change_gpt_weights(gpt_path)


def get_spepc(hps, filename):
    audio = load_audio(filename, int(hps.data.sampling_rate))
    audio = torch.FloatTensor(audio)
    audio_norm = audio
    audio_norm = audio_norm.unsqueeze(0)
    spec = spectrogram_torch(
        audio_norm,
        hps.data.filter_length,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        center=False,
    )
    return spec


dict_language = {
    i18n("中文"): "all_zh",#全部按中文识别
    i18n("英文"): "en",#全部按英文识别#######不变
    i18n("日文"): "all_ja",#全部按日文识别
    i18n("中英混合"): "zh",#按中英混合识别####不变
    i18n("日英混合"): "ja",#按日英混合识别####不变
    i18n("多语种混合"): "auto",#多语种启动切分识别语种
}


def clean_text_inf(text, language):
    phones, word2ph, norm_text = clean_text(text, language)
    phones = cleaned_text_to_sequence(phones)
    return phones, word2ph, norm_text

dtype=torch.float16 if is_half == True else torch.float32
def get_bert_inf(phones, word2ph, norm_text, language):
    language=language.replace("all_","")
    if language == "zh":
        bert = get_bert_feature(norm_text, word2ph).to(device)#.to(dtype)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half == True else torch.float32,
        ).to(device)

    return bert


splits = {"，", "。", "？", "！", ",", ".", "?", "!", "~", ":", "：", "—", "…", }


def get_first(text):
    pattern = "[" + "".join(re.escape(sep) for sep in splits) + "]"
    text = re.split(pattern, text)[0].strip()
    return text


def get_phones_and_bert(text,language):
    if language in {"en","all_zh","all_ja"}:
        language = language.replace("all_","")
        if language == "en":
            LangSegment.setfilters(["en"])
            formattext = " ".join(tmp["text"] for tmp in LangSegment.getTexts(text))
        else:
            # 因无法区别中日文汉字,以用户输入为准
            formattext = text
        while "  " in formattext:
            formattext = formattext.replace("  ", " ")
        phones, word2ph, norm_text = clean_text_inf(formattext, language)
        if language == "zh":
            bert = get_bert_feature(norm_text, word2ph).to(device)
        else:
            bert = torch.zeros(
                (1024, len(phones)),
                dtype=torch.float16 if is_half == True else torch.float32,
            ).to(device)
    elif language in {"zh", "ja","auto"}:
        textlist=[]
        langlist=[]
        LangSegment.setfilters(["zh","ja","en","ko"])
        if language == "auto":
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "ko":
                    langlist.append("zh")
                    textlist.append(tmp["text"])
                else:
                    langlist.append(tmp["lang"])
                    textlist.append(tmp["text"])
        else:
            for tmp in LangSegment.getTexts(text):
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    # 因无法区别中日文汉字,以用户输入为准
                    langlist.append(language)
                textlist.append(tmp["text"])
        print(textlist)
        print(langlist)
        phones_list = []
        bert_list = []
        norm_text_list = []
        for i in range(len(textlist)):
            lang = langlist[i]
            phones, word2ph, norm_text = clean_text_inf(textlist[i], lang)
            bert = get_bert_inf(phones, word2ph, norm_text, lang)
            phones_list.append(phones)
            norm_text_list.append(norm_text)
            bert_list.append(bert)
        bert = torch.cat(bert_list, dim=1)
        phones = sum(phones_list, [])
        norm_text = ''.join(norm_text_list)

    return phones,bert.to(dtype),norm_text


def merge_short_text_in_array(texts, threshold):
    if (len(texts)) < 2:
        return texts
    result = []
    text = ""
    for ele in texts:
        text += ele
        if len(text) >= threshold:
            result.append(text)
            text = ""
    if (len(text) > 0):
        if len(result) == 0:
            result.append(text)
        else:
            result[len(result) - 1] += text
    return result

from scipy.io.wavfile import write

def get_tts_wav(ref_wav_path, prompt_text, prompt_language, text, text_language, save_path, how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free = False):
    if prompt_text is None or len(prompt_text) == 0:
        ref_free = True
    t0 = ttime()
    prompt_language = dict_language[prompt_language]
    text_language = dict_language[text_language]
    if not ref_free:
        prompt_text = prompt_text.strip("\n")
        if (prompt_text[-1] not in splits): prompt_text += "。" if prompt_language != "en" else "."
        print(i18n("实际输入的参考文本:"), prompt_text)
    text = text.strip("\n")
    if (text[0] not in splits and len(get_first(text)) < 4): text = "。" + text if text_language != "en" else "." + text
    
    print(i18n("实际输入的目标文本:"), text)
    zero_wav = np.zeros(
        int(hps.data.sampling_rate * 0.3),
        dtype=np.float16 if is_half == True else np.float32,
    )
    with torch.no_grad():
        wav16k, sr = librosa.load(ref_wav_path, sr=16000)
        if (wav16k.shape[0] > 320000 or wav16k.shape[0] < 0):
            raise OSError(i18n("参考音频在0~20秒范围外，请更换！"))
        wav16k = torch.from_numpy(wav16k)
        zero_wav_torch = torch.from_numpy(zero_wav)
        if is_half == True:
            wav16k = wav16k.half().to(device)
            zero_wav_torch = zero_wav_torch.half().to(device)
        else:
            wav16k = wav16k.to(device)
            zero_wav_torch = zero_wav_torch.to(device)
        wav16k = torch.cat([wav16k, zero_wav_torch])
        ssl_content = ssl_model.model(wav16k.unsqueeze(0))[
            "last_hidden_state"
        ].transpose(
            1, 2
        )  # .float()
        codes = vq_model.extract_latent(ssl_content)
   
        prompt_semantic = codes[0, 0]
    t1 = ttime()

    if (how_to_cut == i18n("凑四句一切")):
        text = cut1(text)
    elif (how_to_cut == i18n("凑50字一切")):
        text = cut2(text)
    elif (how_to_cut == i18n("按中文句号。切")):
        text = cut3(text)
    elif (how_to_cut == i18n("按英文句号.切")):
        text = cut4(text)
    elif (how_to_cut == i18n("按标点符号切")):
        text = cut5(text)
    while "\n\n" in text:
        text = text.replace("\n\n", "\n")
    print(i18n("实际输入的目标文本(切句后):"), text)
    texts = text.split("\n")
    texts = merge_short_text_in_array(texts, 5)
    audio_opt = []
    if not ref_free:
        phones1,bert1,norm_text1=get_phones_and_bert(prompt_text, prompt_language)

    for text in texts:
        # 解决输入目标文本的空行导致报错的问题
        if (len(text.strip()) == 0):
            continue
        if (text[-1] not in splits): text += "。" if text_language != "en" else "."
        print(i18n("实际输入的目标文本(每句):"), text)
        phones2,bert2,norm_text2=get_phones_and_bert(text, text_language)
        print(i18n("前端处理后的文本(每句):"), norm_text2)
        if not ref_free:
            bert = torch.cat([bert1, bert2], 1)
            all_phoneme_ids = torch.LongTensor(phones1+phones2).to(device).unsqueeze(0)
        else:
            bert = bert2
            all_phoneme_ids = torch.LongTensor(phones2).to(device).unsqueeze(0)

        bert = bert.to(device).unsqueeze(0)
        all_phoneme_len = torch.tensor([all_phoneme_ids.shape[-1]]).to(device)
        prompt = prompt_semantic.unsqueeze(0).to(device)
        t2 = ttime()
        with torch.no_grad():
            # pred_semantic = t2s_model.model.infer(
            pred_semantic, idx = t2s_model.model.infer_panel(
                all_phoneme_ids,
                all_phoneme_len,
                None if ref_free else prompt,
                bert,
                # prompt_phone_len=ph_offset,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                early_stop_num=hz * max_sec,
            )
        t3 = ttime()
        # print(pred_semantic.shape,idx)
        pred_semantic = pred_semantic[:, -idx:].unsqueeze(
            0
        )  # .unsqueeze(0)#mq要多unsqueeze一次
        refer = get_spepc(hps, ref_wav_path)  # .to(device)
        if is_half == True:
            refer = refer.half().to(device)
        else:
            refer = refer.to(device)
        # audio = vq_model.decode(pred_semantic, all_phoneme_ids, refer).detach().cpu().numpy()[0, 0]
        audio = (
            vq_model.decode(
                pred_semantic, torch.LongTensor(phones2).to(device).unsqueeze(0), refer
            )
                .detach()
                .cpu()
                .numpy()[0, 0]
        )  ###试试重建不带上prompt部分
        max_audio=np.abs(audio).max()#简单防止16bit爆音
        if max_audio>1:audio/=max_audio
        audio_opt.append(audio)
        audio_opt.append(zero_wav)
        t4 = ttime()
    print("%.3f\t%.3f\t%.3f\t%.3f" % (t1 - t0, t2 - t1, t3 - t2, t4 - t3))

    write(f"output/{save_path}.wav", hps.data.sampling_rate, (np.concatenate(audio_opt, 0) * 32768).astype(np.int16))

    return f"output/{save_path}.wav"

def split(todo_text):
    todo_text = todo_text.replace("……", "。").replace("——", "，")
    if todo_text[-1] not in splits:
        todo_text += "。"
    i_split_head = i_split_tail = 0
    len_text = len(todo_text)
    todo_texts = []
    while 1:
        if i_split_head >= len_text:
            break  # 结尾一定有标点，所以直接跳出即可，最后一段在上次已加入
        if todo_text[i_split_head] in splits:
            i_split_head += 1
            todo_texts.append(todo_text[i_split_tail:i_split_head])
            i_split_tail = i_split_head
        else:
            i_split_head += 1
    return todo_texts


def cut1(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    split_idx = list(range(0, len(inps), 4))
    split_idx[-1] = None
    if len(split_idx) > 1:
        opts = []
        for idx in range(len(split_idx) - 1):
            opts.append("".join(inps[split_idx[idx]: split_idx[idx + 1]]))
    else:
        opts = [inp]
    return "\n".join(opts)


def cut2(inp):
    inp = inp.strip("\n")
    inps = split(inp)
    if len(inps) < 2:
        return inp
    opts = []
    summ = 0
    tmp_str = ""
    for i in range(len(inps)):
        summ += len(inps[i])
        tmp_str += inps[i]
        if summ > 50:
            summ = 0
            opts.append(tmp_str)
            tmp_str = ""
    if tmp_str != "":
        opts.append(tmp_str)
    # print(opts)
    if len(opts) > 1 and len(opts[-1]) < 50:  ##如果最后一个太短了，和前一个合一起
        opts[-2] = opts[-2] + opts[-1]
        opts = opts[:-1]
    return "\n".join(opts)


def cut3(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip("。").split("。")])


def cut4(inp):
    inp = inp.strip("\n")
    return "\n".join(["%s" % item for item in inp.strip(".").split(".")])


# contributed by https://github.com/AI-Hobbyist/GPT-SoVITS/blob/main/GPT_SoVITS/inference_webui.py
def cut5(inp):
    # if not re.search(r'[^\w\s]', inp[-1]):
    # inp += '。'
    inp = inp.strip("\n")
    punds = r'[,.;?!、，。？！;：…]'
    items = re.split(f'({punds})', inp)
    mergeitems = ["".join(group) for group in zip(items[::2], items[1::2])]
    # 在句子不存在符号或句尾无符号的时候保证文本完整
    if len(items)%2 == 1:
        mergeitems.append(items[-1])
    opt = "\n".join(mergeitems)
    return opt


def custom_sort_key(s):
    # 使用正则表达式提取字符串中的数字部分和非数字部分
    parts = re.split('(\d+)', s)
    # 将数字部分转换为整数，非数字部分保持不变
    parts = [int(part) if part.isdigit() else part for part in parts]
    return parts


def change_choices():
    SoVITS_names, GPT_names = get_weights_names()
    return {"choices": sorted(SoVITS_names, key=custom_sort_key), "__type__": "update"}, {"choices": sorted(GPT_names, key=custom_sort_key), "__type__": "update"}


pretrained_sovits_name = "GPT_SoVITS/pretrained_models/s2G488k.pth"
pretrained_gpt_name = "GPT_SoVITS/pretrained_models/s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt"
SoVITS_weight_root = "SoVITS_weights"
GPT_weight_root = "GPT_weights"
os.makedirs(SoVITS_weight_root, exist_ok=True)
os.makedirs(GPT_weight_root, exist_ok=True)


def get_weights_names():
    SoVITS_names = [pretrained_sovits_name]
    for name in os.listdir(SoVITS_weight_root):
        if name.endswith(".pth"): SoVITS_names.append("%s/%s" % (SoVITS_weight_root, name))
    GPT_names = [pretrained_gpt_name]
    for name in os.listdir(GPT_weight_root):
        if name.endswith(".ckpt"): GPT_names.append("%s/%s" % (GPT_weight_root, name))
    return SoVITS_names, GPT_names


SoVITS_names, GPT_names = get_weights_names()

# SRT

class subtitle:
    def __init__(self,index:int, start_time, end_time, text:str):
        self.index = int(index)
        self.start_time = start_time
        self.end_time = end_time
        self.text = text.strip()
    def normalize(self,ntype:str,fps=30):
         if ntype=="prcsv":
              h,m,s,fs=(self.start_time.replace(';',':')).split(":")#seconds
              self.start_time=int(h)*3600+int(m)*60+int(s)+round(int(fs)/fps,2)
              h,m,s,fs=(self.end_time.replace(';',':')).split(":")
              self.end_time=int(h)*3600+int(m)*60+int(s)+round(int(fs)/fps,2)
         elif ntype=="srt":
             h,m,s=self.start_time.split(":")
             s=s.replace(",",".")
             self.start_time=int(h)*3600+int(m)*60+round(float(s),2)
             h,m,s=self.end_time.split(":")
             s=s.replace(",",".")
             self.end_time=int(h)*3600+int(m)*60+round(float(s),2)
         else:
             raise ValueError
    def add_offset(self,offset=0):
        self.start_time+=offset
        if self.start_time<0:
            self.start_time=0
        self.end_time+=offset
        if self.end_time<0:
            self.end_time=0
    def __str__(self) -> str:
        return f'id:{self.index},start:{self.start_time},end:{self.end_time},text:{self.text}'

def read_srt(uploaded_file):
    offset=0
    with open(uploaded_file.name,"r",encoding="utf-8") as f:
        file=f.readlines()
    subtitle_list=[]
    indexlist=[]
    filelength=len(file)
    for i in range(0,filelength):
        if " --> " in file[i]:
            is_st=True
            for char in file[i-1].strip().replace("\ufeff",""):
                if char not in ['0','1','2','3','4','5','6','7','8','9']:
                    is_st=False
                    break
            if is_st:
                indexlist.append(i) #get line id
    listlength=len(indexlist)
    for i in range(0,listlength-1):
        st,et=file[indexlist[i]].split(" --> ")
        id=int(file[indexlist[i]-1].strip().replace("\ufeff",""))
        text=""
        for x in range(indexlist[i]+1,indexlist[i+1]-2):
            text+=file[x]
        st=subtitle(id,st,et,text)
        st.normalize(ntype="srt")
        st.add_offset(offset=offset)
        subtitle_list.append(st)
    st,et=file[indexlist[-1]].split(" --> ")
    id=file[indexlist[-1]-1]
    text=""
    for x in range(indexlist[-1]+1,filelength):
        text+=file[x]
    st=subtitle(id,st,et,text)
    st.normalize(ntype="srt")
    st.add_offset(offset=offset)
    subtitle_list.append(st)
    return subtitle_list

from pydub import AudioSegment

def trim_audio(intervals, input_file_path, output_file_path):
    # load the audio file
    audio = AudioSegment.from_file(input_file_path)

    # iterate over the list of time intervals
    for i, (start_time, end_time) in enumerate(intervals):
        # extract the segment of the audio
        segment = audio[start_time*1000:end_time*1000]

        # construct the output file path
        output_file_path_i = f"{output_file_path}_{i}.wav"

        # export the segment to a file
        segment.export(output_file_path_i, format='wav')

def merge_audios(folder_path):
    output_file = "AI配音版.wav"
    # Get all WAV files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
    # Sort files based on the last digit in their names
    sorted_files = sorted(files, key=lambda x: int(x.split()[-1].split('.')[0][-1]))
    
    # Initialize an empty audio segment
    merged_audio = AudioSegment.empty()
    
    # Loop through each file, in order, and concatenate them
    for file in sorted_files:
        audio = AudioSegment.from_wav(os.path.join(folder_path, file))
        merged_audio += audio
        print(f"Merged: {file}")
    
    # Export the merged audio to a new file
    merged_audio.export(output_file, format="wav")
    return "AI配音版.wav"

def convert_from_srt(filename, video_full, language, split_model, multilingual):
    subtitle_list = read_srt(filename)
    
    if os.path.exists("audio_full.wav"):
        os.remove("audio_full.wav")

    ffmpeg.input(video_full).output("audio_full.wav", ac=2, ar=44100).run()
    
    if split_model=="UVR-HP2":
        pre_fun = pre_fun_hp2
    else:
        pre_fun = pre_fun_hp5

    filename = "output"
    pre_fun._path_audio_("audio_full.wav", f"./denoised/{split_model}/{filename}/", f"./denoised/{split_model}/{filename}/", "wav")
    if os.path.isdir("output"):
        shutil.rmtree("output")
    if multilingual==False:
        for i in subtitle_list:
            os.makedirs("output", exist_ok=True)
            trim_audio([[i.start_time, i.end_time]], f"./denoised/{split_model}/{filename}/vocal_audio_full.wav_10.wav", f"sliced_audio_{i.index}")
            print(f"正在合成第{i.index}条语音")
            print(f"语音内容：{i.text}")
            get_tts_wav(f"sliced_audio_{i.index}_0.wav", "", i18n("中文"), i.text, language, i.text + " " + str(i.index), how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free = True)
    else:
        for i in subtitle_list:
            os.makedirs("output", exist_ok=True)
            trim_audio([[i.start_time, i.end_time]], f"./denoised/{split_model}/{filename}/vocal_audio_full.wav_10.wav", f"sliced_audio_{i.index}")
            print(f"正在合成第{i.index}条语音")
            print(f"语音内容：{i.text.splitlines()[1]}")
            get_tts_wav(f"sliced_audio_{i.index}_0.wav", i.text.splitlines()[0], i18n("中文"), i.text.splitlines()[1], language, i.text.splitlines()[1] + " " + str(i.index), how_to_cut=i18n("不切"), top_k=20, top_p=0.6, temperature=0.6, ref_free = False)
     
    return merge_audios("output")


def convert_from_srt(filename, video_full, language, split_model, multilingual):
    subtitle_list = read_srt(filename)
    
    if os.path.exists("audio_full.wav"):
        os.remove("audio_full.wav")

    ffmpeg.input(video_full).output("audio_full.wav", ac=2, ar=44100).run()
    
    if split_model=="UVR-HP2":
        pre_fun = pre_fun_hp2
    else:
        pre_fun = pre_fun_hp5

    filename = "output"
    pre_fun._path_audio_("audio_full.wav", f"./denoised/{split_model}/{filename}/", f"./denoised/{split_model}/{filename}/", "wav")
    if os.path.isdir("output"):
        shutil.rmtree("output")
    if multilingual==False:
        for i in subtitle_list:
            os.makedirs("output", exist_ok=True)
            trim_audio([[i.start_time, i.end_time]], f"./denoised/{split_model}/{filename}/vocal_audio_full.wav_10.wav", f"sliced_audio_{i.index}")
            print(f"正在合成第{i.index}条语音")
            print(f"语音内容：{i.text}")
            predict(i.text, language, f"sliced_audio_{i.index}_0.wav", i.text + " " + str(i.index))
    else:
        for i in subtitle_list:
            os.makedirs("output", exist_ok=True)
            trim_audio([[i.start_time, i.end_time]], f"./denoised/{split_model}/{filename}/vocal_audio_full.wav_10.wav", f"sliced_audio_{i.index}")
            print(f"正在合成第{i.index}条语音")
            print(f"语音内容：{i.text.splitlines()[1]}")
            predict(i.text.splitlines()[1], language, f"sliced_audio_{i.index}_0.wav", i.text.splitlines()[1] + " " + str(i.index))
     
    return merge_audios("output")


with gr.Blocks() as app:
    gr.Markdown("# <center>🌊💕🎶 XTTS - SRT文件一键AI配音</center>")
    gr.Markdown("### <center>🌟 只需上传SRT文件和原版配音文件即可，每次一集视频AI自动配音！Developed by Kevin Wang </center>")
    with gr.Row():
        with gr.Column():
            inp1 = gr.File(file_count="single", label="请上传一集视频对应的SRT文件")
            inp2 = gr.Video(label="请上传一集包含原声配音的视频", info="需要是.mp4视频文件")
            inp3 = gr.Dropdown(
                label="请选择SRT文件对应的语言",
                choices=[i18n("中文"), i18n("英文"), i18n("日文"), i18n("中英混合"), i18n("日英混合"), i18n("多语种混合")],
                max_choices=1,
                value=i18n("中文"),
            )
            inp4 = gr.Dropdown(label="请选择用于分离伴奏的模型", info="UVR-HP5去除背景音乐效果更好，但会对人声造成一定的损伤", choices=["UVR-HP2", "UVR-HP5"], value="UVR-HP5")
            inp5 = gr.Checkbox(label="SRT文件是否为双语字幕", info="若为双语字幕，请打勾选择（SRT文件中需要先出现中文字幕，后英文字幕；中英字幕各占一行）")
            btn = gr.Button("一键开启AI配音吧💕", variant="primary")
        with gr.Column():
            out1 = gr.Audio(label="为您生成的AI完整配音")

        btn.click(convert_from_srt, [inp1, inp2, inp3, inp4, inp5], [out1])
        
    gr.Markdown("### <center>注意❗：请勿生成会对任何个人或组织造成侵害的内容，请尊重他人的著作权和知识产权。用户对此程序的任何使用行为与程序开发者无关。</center>")
    gr.HTML('''
        <div class="footer">
                    <p>🌊🏞️🎶 - 江水东流急，滔滔无尽声。 明·顾璘
                    </p>
        </div>
    ''')

app.launch(share=True, show_error=True)

