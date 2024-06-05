from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Tokenizer, Wav2Vec2Processor, Wav2Vec2CTCTokenizer


def make_extractor(from_pretrain: bool, pretrainModel_path: str = None):
    """
    生成数据特征提取器，用于将数据转为模型输入
    :param from_pretrain: 是否从原有设定中加载
    :param pretrainModel_path: 原有设定的路径
    :return:
    """
    if from_pretrain:
        if pretrainModel_path is None:
            print("function make_extractor: pretrainModel_path should not be None")
            exit(1)
        return Wav2Vec2FeatureExtractor.from_pretrained(pretrainModel_path)
    else:
        return Wav2Vec2FeatureExtractor(
            feature_size=1, sampling_rate=16_000, padding_value=0.0, do_normalize=True, return_attention_mask=True
        )


def make_tokenizer(from_pretrain: bool, pretrainModel_path: str = None, vocab_json_path: str = None):
    """
    生成tokenizer
    :param from_pretrain: 是否从原有设定中加载
    :param pretrainModel_path: 原有设定的路径
    :param vocab_json_path: 字典的路径
    :return:
    """
    if from_pretrain:
        if pretrainModel_path is None:
            print("function make_tokenizer: pretrainModel_path should not be None")
            exit(1)
        return Wav2Vec2Tokenizer.from_pretrained(pretrainModel_path)
    else:
        if vocab_json_path is None:
            print("function make_tokenizer: vocab_json_path should not be None")
            exit(1)
        # print("medium")

        return Wav2Vec2CTCTokenizer(
            vocab_file=vocab_json_path,
            bos_token="<S>",
            eos_token="</S>",
            unk_token="<SPOKEN_NOISE>",
            pad_token="<PAD>",
            word_delimiter_token="<GAP>",
            replace_word_delimiter_char="",
            return_attention_mask=True
        )


def make_processor(from_pretrain: bool, pretrainModel_path: str = None, vocab_json_path: str = None):
    if from_pretrain:
        if pretrainModel_path is None:
            print("function make_processor: pretrainModel_path should not be None")
            exit(1)
        return Wav2Vec2Processor.from_pretrained(pretrainModel_path)
    else:
        extractor = make_extractor(False)
        tokenizer = make_tokenizer(False, vocab_json_path=vocab_json_path)

        return Wav2Vec2Processor(extractor, tokenizer)


if __name__ == "__main__":
    vocab_json_path = "../../../MySpeech_data_metadata/own_vocab.json"
    save_dir_path = "../../../MySpeech_data_metadata"

    processor = make_processor(False, vocab_json_path=vocab_json_path)
    processor.save_pretrained(save_dir_path)
    print(len(processor.tokenizer))

# def displayWaveform(samples, sr):  # 显示语音时域波形
# 	# samples = samples[6000:16000]
#
# 	print(len(samples), sr)
# 	time = np.arange(0, len(samples)) * (1.0 / sr)
#
# 	plt.plot(time, samples)
# 	plt.title("语音信号时域波形")
# 	plt.xlabel("时长（秒）")
# 	plt.ylabel("振幅")
# 	plt.savefig("./waveform.png")
# 	plt.show()


# if __name__ == "__main__":
# 	path = "/media/mixxis/T7/root/语音/KeSpeech/Audio/1000001/phase1/1000001_63740949.wav"
# 	a_path = "/media/mixxis/T7/root/语音/KeSpeech/Audio/1000002/phase1/1000001_2c863844.wav"
# 	# vad = make_vad()
# 	# wav, _, _ = batch_audio_preprocessing([path, path], vad)
# 	# print(wav.max(), " ", wav.min())
# 	# print(len(wav[0]), len(wav[1]))
# 	# displayWaveform(wav.numpy()[1], 16000)
# 	waveform, sample_rate = torchaudio.load(path)
# 	displayWaveform(waveform.numpy()[0], sample_rate)
