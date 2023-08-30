import torch
from acdc.docstring.utils import AllDataThings
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from acdc.acdc_utils import kl_divergence
from functools import partial



def get_llama2_7b_chat(device="cuda") -> HookedTransformer:
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
    hf_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        low_cpu_mem_usage=True,
    )

    tl_model = HookedTransformer.from_pretrained(
        "meta-llama/Llama-2-7b-chat-hf",
        hf_model=hf_model,
        device="cpu",
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
        tokenizer=tokenizer,
        use_attn_result=True,
        use_split_qkv_input=True,
    )
    tl_model = tl_model.to(device)

    return tl_model


def get_sentences(tl_model: HookedTransformer) -> torch.Tensor:
    df = pd.read_json(
        "https://raw.githubusercontent.com/yanaiela/pararel/"
        "main/data/trex_lms_vocab/P106.jsonl",
        lines=True,
    )
    df["sub_n_tokens"] = [
        len(x) for x in tl_model.tokenizer(tuple(df.sub_label))["input_ids"]
    ]
    df["obj_n_tokens"] = [
        len(x) for x in tl_model.tokenizer(tuple(df.obj_label))["input_ids"]
    ]

    # Has Taku Iwasaki, who wrote Libera Me From Hell
    sub_df = df.query("sub_n_tokens == 6")

    sentences = [
        """\
[INST] <<SYS>>
You are a helpful assistant who responds with one word.
<</SYS>>

What is the profession of {subject}? [/INST]
"""
        for subject in sub_df.sub_label
    ]


def get_all_competing_things() -> AllDataThings:
    tl_model = get_llama2_7b_chat()

    prompt = """\
    [INST] <<SYS>>
    You are a helpful assistant who responds with one word.
    <</SYS>>

    Where is the Eiffel Tower located? [/INST]
    """

    validation_data = torch.LongTensor(tl_model.tokenizer([prompt]).input_ids)

    validation_metric = partial(
        kl_divergence,
        base_model_logprobs=base_validation_logprobs,
        last_seq_element_only=True,
        base_model_probs_last_seq_element_only=False,
        return_one_element=kl_return_one_element,
    )

    return AllDataThings(
        tl_model=tl_model,
        validation_metric=validation_metric,
        validation_data=validation_data,
        validation_labels=validation_labels,
        validation_mask=None,
        validation_patch_data=validation_patch_data,
        test_metrics=test_metrics,
        test_data=test_data,
        test_labels=test_labels,
        test_mask=None,
        test_patch_data=test_patch_data,
    )
