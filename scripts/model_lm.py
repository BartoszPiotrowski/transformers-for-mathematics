import os
import torch
from time import time
from utils import eprint, printf


DEBUG = True
SEP = '='


def train(model, train_data_loader, valid_data_loader, optimizer, tokenizer,
          train_steps_max, save_dir, n=100, N=1000):
    model.train()
    l = len(train_data_loader)
    valid_data_loader_iter = iter(valid_data_loader)
    train_step = 0
    epoch = 0
    while True:
        times, losses = [], []
        for j, batch in enumerate(train_data_loader):
            t0 = time()
            source_target_ids = batch["source_target_ids"].cuda()
            mask              = batch["source_target_mask"].cuda()
            outputs = model(
                input_ids=source_target_ids,
                attention_mask=mask,
                labels=source_target_ids)
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
                singleton_batches = []
                for _ in range(len(source_target_ids)): # train batch size
                    singleton_batches.append(next(valid_data_loader_iter))
                acc = eval_singleton_batches(singleton_batches, model, tokenizer)
                save_path = os.path.join(save_dir, f'model-acc-{acc}')
                #save_path = os.path.join(save_dir, 'model')
                printf(f'saving model to {save_path}')
                model.save_pretrained(save_path)
            train_step += 1
            if train_step > train_steps_max:
                return
        epoch += 1

def eval_singleton_batches(singleton_batches, model, tokenizer,
                           printf_samples=5, testing=False, debug=DEBUG):
    model.eval()
    sources, targets, preds, times = [], [], [], []
    with torch.no_grad():
        for singleton_batch in singleton_batches:
            t0 = time()
            source_ids=singleton_batch['source_ids'].cuda()
            pred_ids=model.generate(
                input_ids=source_ids,
                min_length=1,
                max_length=200,
                num_beams=3) # num_beams=10 seems to cause a memory error
            pred = tokenizer.decode(pred_ids[0], skip_special_tokens=True)
            t1 = time() - t0
            if testing:
                pred_str = pred.split(SEP)[-1].strip(' ')
                eprint(f'eval time (singleton batch): {t1:.3f}')
                printf(pred_str)
            else:
                source = tokenizer.decode(source_ids[0], skip_special_tokens=True)
                target_ids = singleton_batch['target_ids'].cuda()
                target = tokenizer.decode(target_ids[0], skip_special_tokens=True)
                sources.append(source)
                targets.append(target)
                preds.append(pred)
                times.append(t1)
        if testing:
            return
        else:
            avg_eval_time = sum(times) / len(times)
            matching = [1 if p == s + ' ' + t else 0
                        for p, s, t in zip(preds, sources, targets)]
            accuracy = sum(matching) / len(matching)
            printf(f'acc: {accuracy:.3f}, eval time: {avg_eval_time:.3f}\n')
            to_printf = list(reversed(list(zip(sources, targets, preds))))
            if debug:
                source_raw = tokenizer.decode(source_ids[0])
                target_raw = tokenizer.decode(target_ids[0])
                pred_raw = tokenizer.decode(pred_ids[0])
                printf(f'Source + target raw: {source_raw} {target_raw}')
                printf(f'Source + pred raw  : {pred_raw}\n')
            for source, target, pred in to_printf[:printf_samples]:
                printf(f'Source + target: {source} {target}')
                printf(f'Source + pred  : {pred}\n')
            return accuracy


def eval(model, test_data, tokenizer):
    eval_singleton_batches(test_data, model, tokenizer, testing=True)
