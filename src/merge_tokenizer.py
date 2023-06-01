from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import os
import sentencepiece as spm
from dataclasses import dataclass, field
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    LlamaTokenizer,
    HfArgumentParser
)

from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model
import sentencepiece as spm

@dataclass
class SpModelArgument:
    
    original_tokenizer: str = field(
        metadata={
            'required': True,
            "help": "originnal tokenizer path"
        }
    )
    
    sp_model: str = field(
        metadata={
            'required': True,
            "help": "sentencepiece model path"
        }
    )

    save_dir: str = field(
        metadata={
            'required': True,
            "help": "path to save"
        }
    )
    
def merge():
# Refer to Chinese LLaMa

    hf_parser = HfArgumentParser((
        SpModelArgument
    ))
    
    sp_model_args, extra_args = hf_parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    # load original tokenizer
    original_tokenizer = AutoTokenizer.from_pretrained(sp_model_args.original_tokenizer, use_fast=False)
    print(f"tokenizer {sp_model_args.original_tokenizer} is loaded!")
    
    # load sp model
    sp_model = spm.SentencePieceProcessor()
    sp_model.Load(sp_model_args.sp_model)
    print(f"sentencepiece model {sp_model_args.sp_model} is loaded!")
    
    original_tokenizer_sp_model_proto = sp_pb2_model.ModelProto()
    original_tokenizer_sp_model_proto.ParseFromString(original_tokenizer.sp_model.serialized_model_proto())
    sp_model_proto = sp_pb2_model.ModelProto()
    sp_model_proto.ParseFromString(sp_model.serialized_model_proto())
    
    origin_length = len(original_tokenizer_sp_model_proto.pieces)
    origin_token_sets = set(p.piece for p in original_tokenizer_sp_model_proto.pieces)
    for p in tqdm(sp_model_proto.pieces):
        piece = p.piece
        if piece not in origin_token_sets:
            new_piece = sp_pb2_model.ModelProto().SentencePiece()
            new_piece.piece = piece
            new_piece.score = 0
            original_tokenizer_sp_model_proto.pieces.append(new_piece)
            
    print(f"origin tokenizer length {origin_length}")
    print(f"new tokenizer length {len(original_tokenizer_sp_model_proto.pieces)}")

    # output
    sp_save_dir = f'{sp_model_args.save_dir}/sentencepiece/'
    hf_save_dir = f'{sp_model_args.save_dir}/hf/'

    os.makedirs(sp_save_dir,exist_ok=True)
    os.makedirs(hf_save_dir,exist_ok=True)
    
    with open(sp_save_dir+'chinese_guranaco.model', 'wb') as f:
        f.write(original_tokenizer_sp_model_proto.SerializeToString())
        
    tokenizer = LlamaTokenizer(vocab_file=sp_save_dir+'chinese_guranaco.model')
    tokenizer.save_pretrained(hf_save_dir)

    print(f"sentencepiece model & hf tokenizer is saved to {sp_model_args.save_dir}")
    
if __name__ == "__main__":
    merge()