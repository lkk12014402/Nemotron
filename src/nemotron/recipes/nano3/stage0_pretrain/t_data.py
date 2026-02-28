import sys
from pathlib import Path

# 修改成你自己的 Megatron-LM 代码路径

from megatron.core.datasets.indexed_dataset import IndexedDataset

def main():
    # 换成你的 .bin/.idx 前缀
    prefix = "/sdp/lkk/Nemotron/src/nemotron/recipes/nano3/stage0_pretrain/output/stage0_pretrain_tiny/sample-5000/runs/8f511437ed796923_1770982920/datasets/nemotron-math-3/787d5a0269210b03/shard_000000"

    # Megatron 通常传 prefix（不带后缀），它会自动找 .idx/.bin
    dataset = IndexedDataset(prefix)

    print("样本数：", len(dataset))

    # 看第 0 条样本的 token id
    tokens_0 = dataset[0]
    print("第 0 条样本长度：", len(tokens_0))
    print("前 50 个 token：", tokens_0[:50])

    # 随便看几条
    for i in range(3):
        tokens = dataset[i]
        print(f"\n样本 {i}: 长度={len(tokens)}，前 20 个 token={tokens[:20]}")

if __name__ == "__main__":
    main()
