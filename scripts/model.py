import os
import torch
from time import time
from utils import eprint, printf

def eval_batch(batch, model, tokenizer, printf_samples=5, debug=False):
    t0_eval = time()
    model.eval()
    with torch.no_grad():
        source_ids = batch['source_ids'].cuda()
        mask = batch['source_mask'].cuda()
        predicted_ids = model.generate(
            input_ids = source_ids,
            attention_mask = mask,
            min_length=1,
            max_length=100,
            num_beams=3, # 10 seems to cause a memory error
        )
        eval_time = time() - t0_eval
        predicted = [tokenizer.decode(g, skip_special_tokens=True,
                   ) for g in predicted_ids]
        if 'target_ids' in batch:
            target_ids = batch['target_ids'].cuda()
            target = [tokenizer.decode(t, skip_special_tokens=True)
                      for t in target_ids]
            source = [tokenizer.decode(t, skip_special_tokens=True)
                      for t in source_ids]
            matching = [1 if p == t else 0 for p, t in zip(predicted, target)]
            accuracy = sum(matching) / len(matching)
            printf(f'acc: {accuracy:.3f}, eval time: {eval_time:.3f}\n')
            if debug:
                target_raw = [tokenizer.decode(t) for t in target_ids]
                source_raw = [tokenizer.decode(t) for t in source_ids]
                predicted_raw = [tokenizer.decode(g) for g in predicted_ids]
            for i in range(printf_samples):
                printf(f'Source   : {source[i]}')
                printf(f'Target   : {target[i]}')
                printf(f'Predicted: {predicted[i]}\n')
                if debug:
                    printf(f'Source raw   : {source_raw[i]}')
                    printf(f'Target raw   : {target_raw[i]}')
                    printf(f'Predicted raw: {predicted_raw[i]}\n')
            return accuracy
        else: # we are in test decode mode
            for p in predicted:
                printf(p)

def eval(model, test_data, tokenizer):
    for batch in test_data:
        t0 = time()
        eval_batch(batch, model, tokenizer)
        t1 = time() - t0
        n = len(batch["source_ids"])
        avg_time = t1 / n
        eprint(f'avg decode time: {avg_time:.5f}')

def train(model, data_loader, optimizer, tokenizer, train_steps_max,
          save_dir, n=100, N=1000):
    model.train()
    l = len(data_loader)
    train_step = 0
    epoch = 0
    while True:
        times, losses = [], []
        for j, batch in enumerate(data_loader):
            t0 = time()
            source_ids = batch["source_ids"].cuda()
            target_ids = batch["target_ids"].cuda()
            mask       = batch["source_mask"].cuda()
            target_ids[target_ids == tokenizer.pad_token_id] = -100
            outputs = model(
                input_ids=source_ids,
                attention_mask=mask,
                labels=target_ids)
            loss = outputs[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            times.append(time() - t0)
            losses.append(loss)
            if not j % n:
                avg_time = sum(times) / len(times)
                avg_loss = sum(losses) / len(losses)
                times, losses = [], []
                end = '\n' if j % N else ', '
                printf(f'epoch {epoch}, batch {train_step} ({j}/{l}), '
                      f'avg loss {loss:.5f}, avg train time: {avg_time:.3f}',
                      end=end)
            if not j % N:
                acc = eval_batch(batch, model, tokenizer)
                save_path = os.path.join(save_dir, f'model-acc-{acc}')
                printf(f'saving model to {save_path}')
                model.save_pretrained(save_path)
            train_step += 1
            if train_step > train_steps_max:
                return
        epoch += 1

