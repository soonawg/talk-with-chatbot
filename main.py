import speech_recognition as sr  # 음성 인식 및 음성 -> text 에 필요한 패키지
from transformers import AutoModelForCausalLM, AutoTokenizer  # chatbot 모델
import torch
from gtts import gTTS  # google TTS
import playsound as ps  # 음성 재생 패키지 (오류가 나와서 pygame으로 변경)
import tkinter as tk  # GUI Tkinter
from tkinter import *
from tkinter import ttk
import pygame # 음성 재생 패키지
from datetime import datetime  # TTS mp3 파일 고유성을 위하여 사용

def record_text():
    r = sr.Recognizer()  # r에 음성 인식 객체 생성
    with sr.Microphone() as source: # 마이크에서 입력 받기
        speech = r.listen(source)  # 마이크 입력을 sr이 인식

    try:
        audio = r.recognize_google(speech)  # Google web search API를 사용하여 음성인식 | language="ko-KR) 추가로 한국어 인식 가능
        print("Your speech thinks like\n " + audio)
    except sr.UnknownValueError:
        print("Your speech can not understand")
        audio = ""
    except sr.RequestError as e:
        print("Request Error!; {0}".format(e))
        audio = ""
    
    return audio


tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")  # microsoft/DialoGPT-medium 챗봇 모델 사용
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

chat_history_ids = None

def chat_with_model(input_text):
    global chat_history_ids

    # 입력 텍스트를 토큰화하고 모델에 입력
    new_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')

    # 대화 히스토리와 결합하여 모델에 입력
    bot_input_ids = torch.cat([chat_history_ids, new_input_ids], dim=-1) if chat_history_ids is not None else new_input_ids

    # 모델의 출력을 형성
    chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # 모델의 응답을 디코딩
    output_text = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    # bot_input_ids.shape[-1] : bot_input_ids 텐서의 마지막 차원의 길이를 가져온다. 이는 새로운 입력 문장의 길이이다.
    # chat_history_ids[:, bot_input_ids.shape[-1]:] : chat_history_ids 텐서에서 모든행(':')과 bot_input_ids.shape[-1] 이후의 열을 선택한다. 즉, 기존 대화 히스토리와 새로운 입력 문장을 연결한다.
    # [0] : 선택된 첫 번째 요소를 가져온다. 이는 일반적으로 한 번에 하나의 ㅁ누장을 생성하기 때문에 사용된다.

    return output_text

user_input = record_text()
response = chat_with_model(user_input)

def TTS(response):
    output_filename = datetime.now().strftime("voiceEn_%Y%m%d%H%M%S.mp3")  # TTS 파일 고유화
    tts = gTTS(text=response, lang='en')  # gTTS 라이브러리를 사용하여 텍스트를 읽어주는 음성파일 생성.
    tts.save(output_filename)

    pygame.mixer.init()  # pygame을 통해 보이스 파일 재생하기
    pygame.mixer.music.load(output_filename)
    pygame.mixer.music.play()

    print(response)

def record_and_chat():
    user_input = record_text()
    if user_input:
        response = chat_with_model(user_input)
        display_message("User", user_input)
        display_message("Bot", response)
        TTS(response)


def display_message(sender, message):
    output.insert(tk.END, f"{sender}: {message}\n")
    output.see(tk.END)


# GUI

window = tk.Tk()
window.title("챗봇과 대화해보세요!")
window.geometry("640x480")
window.resizable(False, False)

frame = tk.Frame(window)
frame.pack(pady=10)

record_button = tk.Button(frame, text='Record🎤', width=18, height=2, command=record_and_chat)
record_button.pack(side=tk.LEFT, padx=10)

output = tk.Text(window, wrap=tk.WORD, height=20, width=60)
output.pack(padx=10, pady=10)

window.mainloop()