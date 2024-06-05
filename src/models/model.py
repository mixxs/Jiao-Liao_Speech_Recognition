import torch

from src.models.aid.SiameseEcapaTDNN import Siamese_ECAPA_TDNN
from src.models.aid.SiameseWav2vec2AID import Siamese_Wav2Vec2_AID
from src.models.aid.wav2vecAid import Wav2Vec2_AID
from src.models.ctc.wav2vec2CTC import Wav2Vec2_CTC
from src.models.ctc_aid.wav2vec2CTC_AID import Wav2Vec2_CTC_AID
from src.models.ctc_aid.wav2vec2CTC_ecapatdnnAID import Wav2Vec2_CTC_ECAPA_AID


def init_model(model_name: str, processor, model_args, encoder_dir_path: str, model_config,
               language_num: int = 1, aid_model_dir_path: str = None, dialect_feature: torch.Tensor = None):
    if model_name.lower() == "ctc":
        return Wav2Vec2_CTC(model_args=model_args, encoder_dir_path=encoder_dir_path, processor=processor,
                            model_config=model_config)
    elif model_name.lower() == "wav2vec2_ctc_aid":
        return Wav2Vec2_CTC_AID(model_args=model_args, encoder_dir_path=encoder_dir_path, processor=processor,
                                config=model_config, language_num=language_num)
    elif model_name.lower() == "wav2vec2_aid":
        return Wav2Vec2_AID(model_args=model_args, encoder_dir_path=encoder_dir_path,
                            config=model_config, language_num=language_num)
    elif model_name.lower() == "wav2vec2_ctc_ecapatdnn_aid":
        return Wav2Vec2_CTC_ECAPA_AID(model_args=model_args, encoder_dir_path=encoder_dir_path, processor=processor,
                                      config=model_config, ecapa_model_path=aid_model_dir_path,
                                      dialect_feature=dialect_feature)
    elif model_name.lower() == "siamese_wav2vec2_aid":
        return Siamese_Wav2Vec2_AID(model_args=model_args, encoder_dir_path=encoder_dir_path, config=model_config,
                                    language_num=language_num)
    elif model_name.lower() == "siamese_ecapa_tdnn":
        return Siamese_ECAPA_TDNN(encoder_dir_path=encoder_dir_path)
