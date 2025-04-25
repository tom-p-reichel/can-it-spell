from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import numpy as np
import itertools
import more_itertools as miter
import tqdm
import random
from collections import defaultdict

def can_it_spell(model,tok,words):

    conversations = [
        [
            dict(role="user",content="You are in a spelling competition. You have to spell the words you are given. For instance, if you are asked to spell \"mycology\" you must respond with \"M-Y-C-O-L-O-G-Y\". Now, spell \"taxonomy\"."),
            dict(role="assistant",content="T-A-X-O-N-O-M-Y."),
            dict(role="user",content=f"Good. Now spell \"{w}\".")
        ]
        for w in words
    ]
    prompt_lengths = [len(tok.apply_chat_template(c,return_dict=True,add_generation_prompt=True)["input_ids"]) for c in conversations]
    
    for x,w in zip(conversations,words):
        spelt = "-".join(w.upper())+"."
        x.append(
            dict(role="assistant",content=spelt)
        )

    complete_lengths = [len(tok.apply_chat_template(c,return_dict=True,add_generation_prompt=False)["input_ids"]) for c in conversations]
 
    tok.pad_token = tok.eos_token
    
    batch = tok.apply_chat_template(conversations,padding=True,padding_side="left",return_dict=True,add_generation_prompt=False,return_tensors="pt")

    

    with torch.no_grad():
        out = model(**batch.to(model.device))

    # calculate the probability that the model responds with the correct spelling

    # (this doesn't include the probability that the model responds with the correct spelling,
    # but in an alternative format. however, we've prompted it with the right format twice,
    # so it's probably a pretty good proxy for true correctness odds)
    
    probs = torch.nn.functional.softmax(out.logits[:,:-1,:],dim=-1)[
        np.arange(out.logits.shape[0])[:,None],
        np.arange(out.logits.shape[1]-1)[None,:],
        batch["input_ids"][:,1:]
    ]


    correct_prob = []

    for p, pl, cl in zip(probs,prompt_lengths,complete_lengths):
        completion_length = cl-pl

        correct_prob.append(torch.exp(torch.log(p[-completion_length:]).sum()))

    return correct_prob



def can_it_spell_sampled(model,tok,words):

    conversations = [
        [
            dict(role="user",content="You are in a spelling competition. You have to spell the words you are given. For instance, if you are asked to spell \"mycology\" you must respond with \"M-Y-C-O-L-O-G-Y\". Now, spell \"taxonomy\"."),
            dict(role="assistant",content="T-A-X-O-N-O-M-Y."),
            dict(role="user",content=f"Good. Now spell \"{w}\".")
        ]
        for w in words
    ]
    
    batch = tok.apply_chat_template(conversations,padding=True,return_dict=True,add_generation_prompt=True,return_tensors="pt")

    with torch.no_grad():
        out = model.generate(**batch.to(model.device),max_new_tokens=100,do_sample=True)

    prompt_length = batch.input_ids.shape[1]

    return out[:,prompt_length:]


if __name__=="__main__":
    import sqlite3
    
    con = sqlite3.connect("spelling.db")

    cur = con.cursor()

    cur.execute("CREATE TABLE IF NOT EXISTS spelling (model TEXT, word TEXT, prob REAL)")

    quant = BitsAndBytesConfig(load_in_8bit=True,bnb_4bit_compute_dtype=torch.bfloat16)


    models = [
        #dict(pretrained_model_name_or_path="Qwen/Qwen2.5-14B",quantization_config=quant),
        dict(pretrained_model_name_or_path="Qwen/Qwen2-7B-Instruct"),
        dict(pretrained_model_name_or_path="Qwen/Qwen2.5-7B-Instruct"),
        dict(pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.1"),
        dict(pretrained_model_name_or_path="mistralai/Mistral-7B-Instruct-v0.3"),
        dict(pretrained_model_name_or_path="tiiuae/Falcon3-7B-Instruct"),

    ]


    with open("/usr/share/dict/american-english") as f:
        lines = f.readlines()
        words = [x.strip().lower() for x in lines if x.strip().isalpha()]

    scores = []

    word_scores = defaultdict(lambda:0)





    for model_args in models:
        
        words_already_spelled = [x[0] for x in cur.execute("SELECT word FROM spelling WHERE model=? ", (model_args["pretrained_model_name_or_path"],)).fetchall()]

        print(words_already_spelled)

        batched_words_to_spell = list(miter.chunked(set(words)-set(words_already_spelled),32))
        
        score = 0
        model = AutoModelForCausalLM.from_pretrained(**model_args,device_map="cuda",torch_dtype=torch.float16)

        tok = AutoTokenizer.from_pretrained(model_args["pretrained_model_name_or_path"],padding_side="left")

        print(model_args["pretrained_model_name_or_path"])

        for word_batch in tqdm.tqdm(batched_words_to_spell):
            ps = can_it_spell(model,tok,word_batch)

            for w,p in zip(word_batch,ps):
                word_scores[w] += p.item()
                cur.execute("INSERT INTO spelling VALUES (?,?,?)",(model_args["pretrained_model_name_or_path"],w,p.item()))
            
            con.commit()
                
            score += sum(x.item() for x in ps)



        #compare = can_it_spell_sampled(model,tok,words)

        del model


        scores.append(score)
        print(score)


    con.close()


