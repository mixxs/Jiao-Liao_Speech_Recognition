from typing import Union

from transformers import Wav2Vec2Config

from src.models.ctc.wav2vec2CTC import Wav2Vec2_CTC
from src.models.ctc_aid.wav2vec2CTC_AID import Wav2Vec2_CTC_AID
from src.models.ctc_aid.wav2vec2CTC_ecapatdnnAID import Wav2Vec2_CTC_ECAPA_AID
from src.modules.WFAdapter.SequenceAdapterBlock import LinearAdapterBlock, WFAdapterBlock


def add_wf_adapter(
        myWav2vec: Union[Wav2Vec2_CTC, Wav2Vec2_CTC_AID, Wav2Vec2_CTC_ECAPA_AID],
        config: Wav2Vec2Config,
        adapter_embed_dim, language_num, layer_num, freeze: bool = True):
    if freeze:
        for param in myWav2vec.wav2vec2.parameters():
            param.requires_grad = False
    myWav2vec.wav2vec2.adapter = WFAdapterBlock(config=config, adapter_embed_dim=adapter_embed_dim,
                                                language_num=language_num,
                                                layer_num=layer_num)
    for param in myWav2vec.wav2vec2.adapter.parameters():
        param.requires_grad = True
    return myWav2vec
