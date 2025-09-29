import sys
import os
import re
from collections import Counter
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import argparse
import json
from datetime import datetime
import time


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_papers", type=str, help="Path to input papers.json")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs (default=50)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (default=32)")
    return parser.parse_args()


# 1) Text cleaning －－小寫、去除非字母字元（保留空白）、切詞、移除過短詞
def clean_text(text: str) -> List[str]:
    # Convert to lowercase
    # Remove non-alphabetic characters except spaces
    # Split into words
    # Remove very short words (< 2 characters)
    """
    依題目要求清理文字：
      - 全部轉小寫
      - 移除非英文字母（保留空白）
      - 以空白切詞
      - 移除長度 < 2 的短詞
    回傳：乾淨的 tokens list
    """
    if not text:
        return []
    
    text = text.lower()
    # 只保留字母與空白：把不是 a-z 或空白的字元換成空白
    text = re.sub(r"[^a-z\s]", " ", text)

    # 以一個以上空白分割；並過濾過短詞
    words = [t for t in re.split(r"\s+", text) if len(t) >= 2]

    return words


# 2) Vocabulary building －－只保留最常見的前 K 個詞；index 0 保留給 <unk>
def build_vocabulary(
    abstracts: List[str],
    max_words: int = 5000,
    min_freq: int = 1,
) -> Tuple[Dict[str, int], Dict[int, str], Counter]:
    """
    從多篇 abstract 建立詞彙表：
      - 先 clean_text
      - 統計詞頻
      - 只保留出現次數 >= min_freq 的詞
      - 取前 max_vocab 高頻詞（依頻率與字母順序穩定排序）
      - 0 預留給 <unk>
    回傳：(vocab_to_idx, idx_to_vocab, freq_counter)
    """
    freq_words = Counter()
    for abstract in abstracts:
        freq_words.update(clean_text(abstract))

    # 依頻率(降冪)＋詞彙字母序 做穩定排序
    list_of_words = [word for word, count in sorted(freq_words.items(), key=lambda kv: (-kv[1], kv[0])) if count >= min_freq]
    top_5000_words = list_of_words[:max_words] # List[str]

    word_to_idx = {"unknown": 0}
    for idx, word in enumerate(top_5000_words, start=1):
        word_to_idx[word] = idx
    # len of word_to_idx = 5001

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    return word_to_idx, idx_to_word, freq_words


# 3) Sequence encoding －－把文字轉成 index 序列（可 pad/truncate），同時建立 BoW（Autoencoder 輸入/輸出）
def encode_sequences(
    abstracts: List[str],
    word_to_idx: Dict[str, int],
    max_len: int = 150,
) -> Tuple[torch.LongTensor, torch.FloatTensor]:
    """
    將每篇 abstract 轉成：
      - padded/truncated 的 index 序列 (shape: [N, max_len])
      - bag-of-words 向量 (shape: [N, V])，作為 Autoencoder 的輸入與重建目標
    只用 PyTorch，不使用 numpy。
    """
    unknown_idx = word_to_idx.get("unknown", 0)
    len_words = len(word_to_idx)

    seq_list: List[List[int]] = []
    bow_list: List[torch.Tensor] = []

    for abstract in abstracts:
        words = clean_text(abstract)
        idxs = [word_to_idx.get(word, unknown_idx) for word in words] # List[int]

        # --- 序列：pad / truncate ---
        if len(idxs) >= max_len:
            idxs = idxs[:max_len]
        else:
            idxs = idxs + [0] * (max_len - len(idxs))  # 用 0 pad（剛好也是 <unk>）

        seq_list.append(idxs)

        # --- BoW：多標籤一熱向量（浮點型）---
        bow = torch.zeros(len_words, dtype=torch.float32)
        for ix in [word_to_idx.get(word, unknown_idx) for word in words]:
            # 出現即計數（可改為二元 0/1：bow[ix] = 1.0）
            bow[ix] += 1.0
        # 可選：將計數 clip 成 0/1（與 BCE 更搭）
        bow = torch.clamp(bow, max=1.0) # 把 BoW 向量中大於 1 的值全部壓成 1
        bow_list.append(bow)

    seq_tensor = torch.tensor(seq_list, dtype=torch.long)          # [N, max_len]
    bow_tensor = torch.stack(bow_list, dim=0)                      # [N, V], V = len(word_to_idx)
    
    return seq_tensor, bow_tensor




class TextAutoencoder(nn.Module):

    def __init__(self, vocab_size, hidden_dim, embedding_dim):
        super().__init__()
        # Encoder: vocab_size → hidden_dim → embedding_dim
        self.encoder = nn.Sequential(
            nn.Linear(vocab_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Decoder: embedding_dim → hidden_dim → vocab_size  
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
            nn.Sigmoid()  # Output probabilities
        )
    
    def forward(self, x):
        # Encode to bottleneck
        embedding = self.encoder(x)
        # Decode back to vocabulary space
        reconstruction = self.decoder(embedding)
        return reconstruction, embedding
    

# ---- Helper: count parameters (確認 ≤ 2,000,000) ----
def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():

    # Check command line arguments is correct 
    '''
    if len(sys.argv) != 3:
        print("python train_embeddings.py <input_papers.json> <output_dir>")
        sys.exit(1)

    papers = sys.argv[1]
    output_dir = sys.argv[2]
    '''

    

    args = parse_args()
    input_file = args.input_papers
    output_dir = args.output_dir
    num_epochs = args.epochs
    batch_size = args.batch_size

    # Check the output dirctory exists
    os.makedirs(output_dir, exist_ok=True)


    print("Loading abstracts from papers.json...")
    with open(input_file, "r", encoding="utf-8") as f:
        papers = json.load(f)
    print(f"Found {len(papers)} abstracts")
    arxiv_ids = [p["arxiv_id"] for p in papers]
    abstracts = [p["abstract"] for p in papers]
    

    # Data Preprocessing
    word_to_idx, idx_to_word, freq_words = build_vocabulary(abstracts)
    print(f"Building vocabulary from {len(freq_words)} words...")
    print(f"Vocabulary size: {len(word_to_idx)} words")

    vocabulary = {
        "vocab_to_idx": word_to_idx,
        "idx_to_vocab": idx_to_word,
        "vocab_size": len(word_to_idx),
        "total_words": len(freq_words)
    }

    vocabulary_path = os.path.join(output_dir, "vocabulary.json")
    with open(vocabulary_path, "w", encoding="utf-8") as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)
    
    print(f"vocabulary saved to {vocabulary_path}")

    seq_tensor, bow_tensor = encode_sequences(abstracts, word_to_idx)



    # Training
    vocab_size = len(word_to_idx)     # 你的 V
    hidden_dim = 256
    embedding_dim = 64

    dataset = TensorDataset(bow_tensor, bow_tensor)   # X=Y
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextAutoencoder(vocab_size, hidden_dim, embedding_dim).to(device)
    print(f"Model architecture: {vocab_size} → {hidden_dim} → {embedding_dim} → {hidden_dim} → {vocab_size}")
    # print(model)
    print("Total parameters:", count_trainable_parameters(model))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCELoss()


    

    loss_each_epoch = []

    print("Training autoencoder...")
    t0 = time.time()
    start_dt = datetime.now()
    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0


        for batch_x, batch_y in tqdm(loader, desc=f"Epoch {epoch}", leave=False): 
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            recon, emb = model(batch_x)        # recon: [B,V]
            
            loss = criterion(recon, batch_y)   # 目標是重建輸入
            
            loss.backward()
            optimizer.step()
            
            running += loss.item() * batch_x.size(0)

        avg = running / len(loader.dataset)
        print(f"Epoch {epoch:02d}/{num_epochs}, Loss: {avg:.4f}")

        loss_each_epoch.append(avg)

    t1 = time.time()
    end_dt = datetime.now()
    print(f"Training complete in {t1 - t0:.2f} seconds")


    model_path = os.path.join(output_dir, "model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),   # 模型參數
        'vocab_to_idx': word_to_idx,             # 字典映射，方便未來轉換
        'model_config': {                         # 模型超參數
            'vocab_size': vocab_size,
            'hidden_dim': hidden_dim,
            'embedding_dim': embedding_dim
        }
    }, model_path)

    print(f"Model saved to {model_path}")
    

    training_log = {
        "start_time": start_dt.isoformat(timespec="seconds"),
        "end_time": end_dt.isoformat(timespec="seconds"),
        "epochs": num_epochs,
        "final_loss": loss_each_epoch[-1],
        "total_parameters": count_trainable_parameters(model),
        "papers_processed": len(papers),
        "embedding_dimension": embedding_dim
    }
    training_log_path = os.path.join(output_dir, "training_log.json")
    with open(training_log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, ensure_ascii=False, indent=2)

    print(f"training_log saved to {training_log_path}")



    # === 最後輸出 embeddings + loss ===
    model.eval()

    embeddings = []

    with torch.no_grad():
        for i, bow in enumerate(bow_tensor):
            bow = bow.unsqueeze(0).to(device)         # [1, V]
            recon, emb = model(bow)                   # recon: [1, V], emb: [1, E]
            loss = criterion(recon, bow).item()       # reconstruction loss (單篇)

            embeddings.append({
                "arxiv_id": papers[i].get("arxiv_id"),
                "embedding": emb.squeeze(0).cpu().tolist(),
                "reconstruction_loss": loss
            })

    embeddings_path = os.path.join(output_dir, "embeddings.json")
    with open(embeddings_path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f, ensure_ascii=False, indent=2)

    print(f"Embeddings saved to {embeddings_path}")






if __name__ == "__main__":
    main()