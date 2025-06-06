import argparse

import lightning as lt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm, trange
from transformers import AutoModel, AutoTokenizer


class EmbeddingDataset(Dataset):
    def __init__(self, bert_embeddings, llm_embeddings):
        super().__init__()
        self.bert_embeddings = bert_embeddings
        self.llm_embeddings = llm_embeddings

    def __len__(self):
        return len(self.bert_embeddings)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.bert_embeddings[idx], dtype=torch.float32),
            torch.tensor(self.llm_embeddings[idx], dtype=torch.float32),
        )


class EmbeddingDataModule(lt.LightningDataModule):
    def __init__(
        self,
        bert_embeddings_train,
        llm_embeddings_train,
        bert_embeddings_val,
        llm_embeddings_val,
        batch_size=64,
    ):
        super().__init__()
        self.train_dataset = EmbeddingDataset(
            bert_embeddings_train, llm_embeddings_train
        )
        self.val_dataset = EmbeddingDataset(bert_embeddings_val, llm_embeddings_val)
        self.batch_size = batch_size
        self.llm_shape = llm_embeddings_train.shape[1]
        self.ret_shape = bert_embeddings_train.shape[1]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        raise NotImplementedError

class Autoencoder(lt.LightningModule):
    def __init__(self, llm_shape, ret_shape, llm_loss_alpha=0.25, lr=1e-5):
        super(Autoencoder, self).__init__()
        self.llm_shape = llm_shape
        self.ret_shape = ret_shape
        self.lr = lr
        self.llm_loss_alpha = llm_loss_alpha

        if llm_shape < 2048:
            raise RuntimeWarning("llm shape is below first linear output.")
        if ret_shape > 1024:
            raise RuntimeWarning("ret shape is above first linear output.")

        # Encoder: 4096 -> 768
        self.encoder = nn.Sequential(
            nn.Linear(llm_shape, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, ret_shape),
        )
        # Decoder: 768 -> 4096
        self.decoder = nn.Sequential(
            nn.Linear(ret_shape, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, llm_shape),
        )
        self.criteria = nn.MSELoss()

    def forward_encoder(self, x):
        encoded = self.encoder(x)
        return encoded

    def forward_decoder(self, x):
        decoded = self.decoder(x)
        return decoded

    def training_step(self, batch, batch_idx):
        bert, llm = batch

        encoded = self.forward_encoder(llm)
        decoded = self.forward_decoder(encoded)

        llm_loss = torch.sqrt(self.criteria(decoded, llm))
        bert_loss = torch.sqrt(self.criteria(encoded, bert))
        loss = self.llm_loss_alpha * llm_loss + (1 - self.llm_loss_alpha) * bert_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/llm_loss", llm_loss, prog_bar=True)
        self.log("train/bert_loss", bert_loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        bert, llm = batch

        encoded = self.forward_encoder(llm)
        decoded = self.forward_decoder(encoded)

        llm_loss = torch.sqrt(self.criteria(decoded, llm))
        bert_loss = torch.sqrt(self.criteria(encoded, bert))
        loss = self.llm_loss_alpha * llm_loss + (1 - self.llm_loss_alpha) * bert_loss

        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/llm_loss", llm_loss, prog_bar=True)
        self.log("val/bert_loss", bert_loss, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2
        )
        return [optimizer], [scheduler]


if __name__ == "__main__":

    def parse_args():
        parser = argparse.ArgumentParser(description="Tool Attack Model Configuration")
        parser.add_argument(
            "--device",
            type=str,
            default="cuda",
            help="The device to use for computation",
        )
        parser.add_argument(
            "--ret_model_path",
            type=str,
            default="/newdisk/public/tool_attack_models/contriever-msmarco",
            help="Path to the retriever model",
        )
        parser.add_argument(
            "--llm_model_path",
            type=str,
            default="/newdisk/public/models/Meta-Llama-3-8B-Instruct",
            help="Path to the LLM model",
        )
        parser.add_argument(
            "--output_file_path",
            type=str,
            default="/newdisk/public/tool_attack/ae2/test_contriever_llama3_1",
            help="Path to save the model",
        )
        return parser.parse_args()

    args = parse_args()
    device = torch.device(args.device)

    # %%
    # 加载模型和Tokenizer
    ret_tokenizer = AutoTokenizer.from_pretrained(args.ret_model_path, use_fast=True)
    ret_model = AutoModel.from_pretrained(args.ret_model_path, device_map=args.device)
    llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path, use_fast=True)
    llm_model = AutoModel.from_pretrained(args.llm_model_path, device_map=args.device)

    # %%
    llm_tokens = list(llm_tokenizer.get_vocab().keys())
    ret_tokens = list(ret_tokenizer.get_vocab().keys())
    import copy

    def tiktoken_bpe_2_wordpiece(token):
        if token.startswith("Ġ"):
            token = token.replace("Ġ", "")
            return token
        else:
            token = "##" + token
            return token

    def wordpiece_2_tiktoken_bpe(token):
        if token.startswith("##"):
            token = token.replace("##", "")
            return token
        else:
            token = "Ġ" + token
            return token

    common_tokens = set(llm_tokens).intersection(
        set([wordpiece_2_tiktoken_bpe(ret_token) for ret_token in ret_tokens])
    )
    common_tokens = list(common_tokens)
    common_tokens = list(common_tokens)
    common_tokens_ret = [
        tiktoken_bpe_2_wordpiece(common_token) for common_token in common_tokens
    ]
    print(f"Common Tokens: {len(common_tokens)}")

    common_input_ids_llm = llm_tokenizer.convert_tokens_to_ids(common_tokens)
    common_input_ids_ret = ret_tokenizer.convert_tokens_to_ids(common_tokens_ret)

    llm_embeddings = llm_model.embed_tokens(
        torch.tensor(common_input_ids_llm).to(device)
    )
    ret_embeddings = ret_model.embeddings.word_embeddings(
        torch.tensor(common_input_ids_ret).to(device)
    )

    # 已经检查，确实没有Norm
    ret_emb = ret_embeddings.cpu().detach().numpy()
    llm_emb = llm_embeddings.cpu().detach().numpy()

    del (
        llm_embeddings,
        ret_embeddings,
        common_input_ids_llm,
        common_input_ids_ret,
        ret_tokenizer,
        llm_tokenizer,
        ret_model,
        llm_model,
    )
    torch.cuda.empty_cache()
    print("Embedding Shape:", llm_emb.shape, ret_emb.shape)

    # %%
    import gc

    gc.collect()
    X_train, X_test, y_train, y_test = train_test_split(
        llm_emb, ret_emb, train_size=0.9, random_state=42, shuffle=True
    )

    # %%
    X_train.shape, X_test.shape, y_train.shape, y_test.shape

    # %% [markdown]

    # %%
    def get_statistic_info(y_predict, y_true):
        dis = euclidean_distances(y_predict, y_true)
        series = pd.Series(dis.diagonal())
        return series.describe()

    def count_distance_ratio(y_true, seperator):
        count_closer = (
            np.sum(euclidean_distances(y_true, y_true) <= seperator) - y_true.shape[0]
        )
        count_farther = np.sum(euclidean_distances(y_true, y_true) > seperator)
        return (
            count_closer,
            count_farther,
            count_farther / (count_closer + count_farther),
        )

    def cal_topk(total, min_dis_idx, topks):
        cnts = [0] * len(topks)
        for i in range(total):
            for idx, topk in enumerate(topks):
                if i in min_dis_idx[:topk, i]:
                    cnts[idx] += 1
        return [cnt / total for cnt in cnts]

    def print_to_md(data):
        for item in data:
            print("{:.3f}".format(item * 100), end="|")
        print()

    def metrics(_y_predict_train, _y_predict_test, _y_train, _y_test):
        statistics_train = get_statistic_info(_y_predict_train, _y_train)
        print("# Train Statistics")
        print(statistics_train)
        print("# Test Statistics")
        statistics_test = get_statistic_info(_y_predict_test, _y_test)
        print(statistics_test)
        print("# Distance Ratio")
        print(
            f"All Token Mapping Closer than {100 * count_distance_ratio(_y_train, statistics_test["max"])[2]}% of Original Tokens"
        )
        print(
            f"Averge Token Mapping Closer than {100 * count_distance_ratio(_y_train, statistics_test["mean"])[2]}% of Original Tokens"
        )
        min_dis_idx_test = euclidean_distances(
            _y_predict_test, np.concatenate((_y_test, _y_train), axis=0)
        ).argsort(axis=0)
        min_dis_idx_train = euclidean_distances(
            _y_predict_train, np.concatenate((_y_train, _y_test), axis=0)
        ).argsort(axis=0)
        topk_test = cal_topk(
            _y_test.shape[0], min_dis_idx_test, [1, 3, 5, 10, 20, 50, 100, 200]
        )
        topk_train = cal_topk(
            _y_train.shape[0], min_dis_idx_train, [1, 3, 5, 10, 20, 50, 100, 200]
        )
        print("# Train Target Token in TopK")
        print_to_md(topk_train)
        print("# Test Target Token in TopK")
        print_to_md(topk_test)

    # %% [markdown]
    # #### Part3 AutoEncoder & Lightning

    # %%
    data_module = EmbeddingDataModule(y_train, X_train, y_test, X_test)

    # %%
    model = Autoencoder(
        llm_shape=data_module.llm_shape, ret_shape=data_module.ret_shape
    )
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    lt.seed_everything(42)
    torch.set_float32_matmul_precision("highest")

    callbacks = [
        ModelCheckpoint(
            monitor="val_loss",
            filename="autoencoder-{epoch:02d}-{val_loss:.4f}",
            dirpath=args.output_file_path,
            save_top_k=1,
            save_last=True,
            mode="min",
        ),
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
    ]

    trainer = lt.Trainer(max_epochs=500, callbacks=callbacks)
    trainer.fit(model, datamodule=data_module)

    # %%
    trainer.checkpoint_callback.best_model_path

    # %%
    ae = Autoencoder.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path,
        llm_shape=data_module.llm_shape,
        ret_shape=data_module.ret_shape,
    ).to(device)
    ae.eval()
    ae

    # %%
    y_predict_train = (
        ae.forward_encoder(torch.tensor(X_train).to(ae.device)).cpu().detach().numpy()
    )
    y_predict_test = (
        ae.forward_encoder(torch.tensor(X_test).to(ae.device)).cpu().detach().numpy()
    )

    print(metrics(y_predict_train, y_predict_test, y_train, y_test))
