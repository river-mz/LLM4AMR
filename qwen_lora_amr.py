
# -*- coding: utf-8 -*-
import json
import pandas as pd
import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import os
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
import datetime
import numpy as np
import pandas as pd

# 读取 CSV 数据
data = pd.read_csv("/root/AMR/LLM4AMR/ver1/myData.csv", nrows = 500) # to edit

# 将特征和标签分开
features = data.drop(columns=['Unnamed: 0',"resistance_nitrofurantoin", "resistance_sulfamethoxazole", "resistance_ciprofloxacin", "resistance_levofloxacin"])
labels = data[["resistance_nitrofurantoin", "resistance_sulfamethoxazole", "resistance_ciprofloxacin", "resistance_levofloxacin"]]

# 将特征转换为字符串格式，作为模型的输入

data["input"] = features.apply(lambda row: "; ".join([f"{col.replace('_',' ')}:{val}" for col, val in row.items()]), axis=1)


def process_func(example, antibiotics):
    MAX_LENGTH = 384
    input_ids, attention_mask, labels = [], [], []
    
    # 将特征转换为字符串
    feature_str = example["input"]
    
    # 构建指令
    instruction = tokenizer(
        f"<|im_start|>system\nYou are an expert in predicting antibiotic resistance for {antibiotics} based on patient data. Please output the prediction results.<|im_end|>\n<|im_start|>user\n{feature_str}<|im_end|>\n<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    
    # 构建标签
    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    
    # 合并输入和标签
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 截断到最大长度
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def predict(messages, model, tokenizer):
    device = "cuda"
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# Download Qwen model
snapshot_download("qwen/Qwen2-1.5B-Instruct", cache_dir="./", revision="master")
# Loading Transformers model weight
tokenizer = AutoTokenizer.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", use_fast=False, trust_remote_code=True)

#AutoModelForSequenceClassification
Qwen_model = AutoModelForCausalLM.from_pretrained("./qwen/Qwen2-1___5B-Instruct/", device_map="cuda", torch_dtype=torch.bfloat16)
Qwen_model.enable_input_require_grads()
#loading datasets
fold_k = 4
X = []
Y = []
kf = StratifiedKFold(n_splits=fold_k, shuffle=True,random_state=0)
# txt_add = "./output_my/"+ antimicrobial +"_"+str(feature_num) # to edit 
txt_add = "./output_my_test/"

if not os.path.exists(txt_add):
    os.makedirs(txt_add)



# labels to predict
labels_list = ["resistance_nitrofurantoin", "resistance_sulfamethoxazole", "resistance_ciprofloxacin", "resistance_levofloxacin"]

for label in labels_list:

    X = data['input'].tolist()
    Y = labels[label].tolist()

    valid_indices = [i for i, y in enumerate(Y) if not np.isnan(y)]

    X = [X[i] for i in valid_indices]
    Y = [Y[i] for i in valid_indices]

    antimicrobial = label.split('_')[-1]


    print(f"Training and evaluating for label: {label}")
    
    running_times = 0
    for train_index, test_index in kf.split(X,Y):
        running_times = running_times + 1
        print("Running times:" + str(running_times))
        
        x_train = [X[i] for i in train_index]
        y_train = [Y[i] for i in train_index]
        x_test = [X[i] for i in test_index]
        y_test = [Y[i] for i in test_index]

        #trainning dataset
        len_train = len(train_index)
        print("Trainning datasets length:" + str(len_train))
        messages = []
        for i in range(0,len_train,1):
            feature = x_train[i]
            label = y_train[i]
            message = { # to edit
                    "instruction": f"You are an expert in prediction of antimicrobial resistance for {antimicrobial}, and you will receive gene feature sequences. Please output the prediction results.",
                    # "input":antimicrobial +": "+feature,
                    "input": feature,
                    "output": label,
                }
            messages.append(message)
        train_df = pd.DataFrame(messages)
        train_ds = Dataset.from_pandas(train_df)
        train_dataset = train_ds.map(process_func, fn_kwargs={"antibiotics": antimicrobial}, remove_columns=train_ds.column_names)

        # config Lora model 1.5B
        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            ],
            inference_mode=False,
            r=32,
            lora_alpha=80,
        )
        
        # Qwen + Lora
        model = get_peft_model(Qwen_model, config)

        # Trainning config
        args = TrainingArguments(
            output_dir="./output_my_training/"+ antimicrobial + "/"+datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'),
            per_device_train_batch_size=32,
            gradient_accumulation_steps=2,
            logging_steps=5,
            num_train_epochs=10,
            learning_rate=1e-5,
            save_on_each_node=True,
            gradient_checkpointing=True, # time change spece
            report_to="none",
            lr_scheduler_type = "polynomial",
            seed = 0,
        )

        #set trainer
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
        )
        trainer.train()

        # Test dataset
        len_test = len(test_index)
        print("Test datasets length:" + str(len_test))
        y_label = []
        y_pred = []
        for i in range(0,len_test,1): 
            instruction = f"You are an expert in prediction of antimicrobial resistance for {antimicrobial}, and you will receive patients' electronic health records. Please output the prediction results."
            # feature = antimicrobial +": " + x_test[i]
            feature = x_test[i]
            message = [
                {"role": "system", "content": f"{instruction}"},
                {"role": "user", "content": f"{feature}"},
            ]
            response = predict(message, model, tokenizer)
            y_pred.append(int(float(response)))
            y_label.append(int(float(y_test[i])))

        # Statistics
        cm = confusion_matrix(y_label,y_pred)
        TP = cm[1][1]
        TN = cm[0][0]
        FP = cm[0][1]
        FN = cm[1][0]
        pre = (TP)/(TP+FP)
        rec = (TP)/(TP+FN)
        f1 = 2 * pre * rec / (pre + rec)
        out = "pre:"+str(pre) + " rec:" +str(rec) + " f1:"+str(f1)
        with open(txt_add + f"/{antimicrobial}"+ " output.txt","a") as file:
            file.write(out+'\n')




