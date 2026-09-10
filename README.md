# rag-eval-harness

**⚠️ Demonstration repository using synthetic data.**
**⚠️ 本リポジトリは完全架空データを用いたデモンストレーション用リポジトリです。**

A small, dependency-light harness for evaluating a RAG (Retrieval-Augmented
Generation) retrieval step against a fixed evaluation set, with statistically
sound Before/After comparison.

固定の評価セットに対してRAG(検索拡張生成)の検索精度を測定し、統計的に妥当な
Before/After比較を行うための、依存の少ない軽量な評価ハーネスです。

---

## English

### Purpose

This repository demonstrates the structure of a reusable RAG evaluation
harness:

- a **fixed evaluation set** of questions with known relevant documents,
- standard retrieval **metrics** (Hit@K, MRR, Recall@K, Precision@K),
- **statistics** for comparing two runs (bootstrap confidence intervals, a
  paired permutation test, and Cohen's d effect size), and
- a **report** generator that turns a Before/After comparison into a
  Markdown table.

All data in this repository — the corpus, the questions, and the fictional
product ("ArcLedger", a cloud bookkeeping SaaS) — is **entirely invented**
for this demonstration. It contains no real customer names, store names,
property names, or content derived from any real project or engagement.

### Architecture

```
                 ┌─────────────────────┐
                 │  data/eval_set.json │  30 fixed synthetic questions
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  data/corpus/*.md   │  17 fictional documents
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │  harness/runner.py  │  retriever (mock by default) -> ranked doc IDs
                 └──────────┬──────────┘
                            │
                 ┌──────────▼──────────┐
                 │ harness/metrics.py  │  Hit@K, MRR, Recall@K, Precision@K
                 └──────────┬──────────┘
                            │
        ┌───────────────────┴───────────────────┐
        │                                        │
┌───────▼────────┐                     ┌─────────▼─────────┐
│ harness/stats.py│  CI, permutation   │ harness/report.py │  Before/After
│  test, Cohen's d│                     │  Markdown / JSON   │
└─────────────────┘                     └────────────────────┘
```

Two retrievers ship offline, with no network calls and fully deterministic
output given a fixed seed:

- `naive` — unweighted character-bigram overlap. A stand-in "before" baseline.
- `tfidf` — TF-IDF weighted cosine similarity over the same bigrams. A
  stand-in "after" improvement.

A third, **optional** retriever (`openai`) calls the real OpenAI embeddings
API. It is never used unless explicitly requested with `--mode openai`, and
requires an API key you provide yourself (see [Configuration](#configuration)).
LLM/embedding calls are mocked by default; real API usage is opt-in.

### Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync            # installs the harness + pytest (dev group)
uv sync --extra real   # also installs the optional `openai` dependency
```

### Running the evaluation

```bash
# Run the fixed eval set with the offline TF-IDF retriever
uv run python -m harness.runner --mode tfidf

# Compare the naive baseline against the TF-IDF retriever
uv run python -m harness.report --before naive --after tfidf
```

`runner.py` prints per-query and aggregate metrics as JSON.
`report.py` prints a Markdown table comparing two runs, including confidence
intervals, a permutation-test p-value, and Cohen's d for each metric.

### Configuration

Copy `.env.example` to `.env` only if you intend to use `--mode openai`:

```bash
cp .env.example .env
# fill in OPENAI_API_KEY
```

`.env` is git-ignored and is never required for the default mock/offline path.

### Evaluation methodology

- **Hit@K**: did at least one relevant document appear in the top K results?
- **MRR** (Mean Reciprocal Rank): the average of `1 / rank` of the first
  relevant document across all queries.
- **Recall@K**: fraction of all relevant documents for a query found in the
  top K. Undefined (excluded from the mean, not scored as 0) for queries
  with no relevant documents at all.
- **Precision@K**: fraction of the top K retrieved documents that are
  relevant.
- **Before/After comparison**: a percentile bootstrap gives a confidence
  interval for each run's mean; a paired sign-flip permutation test gives a
  p-value for whether the difference is distinguishable from sampling noise;
  Cohen's d (paired) gives a standardized effect size.

The eval set intentionally includes both single-document questions,
multi-document ("cross_doc") questions, and the metrics are defined to
degrade gracefully on out-of-scope queries — see `tests/` for the boundary
cases exercised (empty relevant sets, k larger than the retrieved list,
identical before/after runs, etc.).

### Sample result summary

Running `uv run python -m harness.report --before naive --after tfidf` over
the shipped 30-question eval set (16 topical documents plus one distractor
FAQ page full of generic boilerplate) shows a small, directionally
consistent improvement from the naive baseline to the TF-IDF retriever —
e.g. MRR 0.983 -> 1.000 and Hit@1 0.967 -> 1.000 — because IDF weighting
lets TF-IDF ignore the boilerplate distractor's repeated common phrases,
while the naive unweighted-overlap baseline is occasionally misled by them.

At this eval set's size (n=30, a single query flips per metric), the
permutation test correctly reports this improvement as **not** statistically
distinguishable from sampling noise (p ≈ 1.0). This is intentional and left
unadjusted: it demonstrates that the harness will not manufacture false
confidence from a small sample, and that "the after run scored higher" is
not by itself evidence of a real improvement. Exact numbers depend on the
environment and are reproducible given the fixed seed (`--seed`, default
`42`) — run the command yourself rather than trusting numbers pasted into a
README.

### Tests

```bash
uv run pytest
```

Covers normal, abnormal, and boundary cases for `metrics.py`, `stats.py`,
`runner.py`, and `report.py`, plus an integration test that loads the
shipped `data/` files end-to-end.

### License

No license is granted — see the notice at the end of this file.

---

## 日本語

### 目的

本リポジトリは、再利用可能なRAG評価ハーネスの構成を示すデモンストレーションです。

- 既知の正解文書付きの**固定評価セット**
- 標準的な検索**評価指標**(Hit@K、MRR、Recall@K、Precision@K)
- 2つの実行結果を比較するための**統計処理**(ブートストラップ信頼区間、
  対応のある並べ替え検定、Cohenのdによる効果量)
- Before/After比較をMarkdown表にまとめる**レポート**生成

本リポジトリに含まれるデータ(コーパス、質問セット、架空の製品
「ArcLedger」というクラウド会計SaaS)は、すべてこのデモのために**新規に
作成した完全架空のもの**です。実在の顧客名・店舗名・物件名、および実案件
由来の内容は一切含まれていません。

### 構成図

上記「Architecture」の図を参照してください(日英共通の構成です)。

オフラインで動作する2種類のリトリーバーを標準搭載しており、外部通信なし・
固定シードによる完全な再現性があります。

- `naive` — 文字バイグラムの単純一致数のみを使う、素朴な「Before」相当のベースライン
- `tfidf` — 同じバイグラムに対してTF-IDF重み付きコサイン類似度を使う、「After」相当の改善版

3つ目の**任意機能**として `openai` モードがあり、実際のOpenAI Embeddings
APIを呼び出せます。`--mode openai` を明示的に指定しない限り使用されず、
利用にはご自身で用意したAPIキーが必要です(詳細は下記「設定」を参照)。
LLM・埋め込み呼び出しは既定でモックであり、実API利用は任意機能です。

### セットアップ

Python 3.11以上と [uv](https://docs.astral.sh/uv/) が必要です。

```bash
uv sync                  # ハーネス本体とpytest(dev group)をインストール
uv sync --extra real     # 任意のopenai依存も追加インストール
```

### 評価の実行

```bash
# オフラインのTF-IDFリトリーバーで固定評価セットを実行
uv run python -m harness.runner --mode tfidf

# naiveベースラインとTF-IDFリトリーバーをBefore/After比較
uv run python -m harness.report --before naive --after tfidf
```

`runner.py` はクエリごとおよび全体集計の評価指標をJSONで出力します。
`report.py` は2つの実行結果を比較したMarkdown表(信頼区間・並べ替え検定の
p値・Cohenのdを含む)を出力します。

### 設定

`--mode openai` を使う場合のみ、`.env.example` を `.env` にコピーしてください。

```bash
cp .env.example .env
# OPENAI_API_KEY を記入
```

`.env` はgit管理対象外であり、既定のモック/オフライン経路では不要です。

### 評価方法

- **Hit@K**: 上位K件の中に正解文書が1件でも含まれているか
- **MRR**(平均逆順位): 各クエリで最初に見つかった正解文書の `1/順位` の平均
- **Recall@K**: そのクエリの全正解文書のうち、上位K件で見つかった割合。
  正解文書が1件もないクエリでは「未定義」として平均計算から除外し、0点
  として扱いません。
- **Precision@K**: 上位K件のうち正解文書である割合
- **Before/After比較**: パーセンタイル・ブートストラップで各実行の平均値の
  信頼区間を、対応のある符号反転並べ替え検定で差が偶然のばらつきと区別
  できるかのp値を、Cohenのd(対応あり)で標準化された効果量を算出します。

評価セットには単一文書で答えられる質問と複数文書にまたがる質問
(`cross_doc`)の両方を含み、各評価指標は範囲外の質問に対しても破綻せずに
動作するよう設計されています。境界値のテストケース(正解文書が0件、kが
検索結果件数より大きい、Before/Afterが完全に同一、など)は `tests/` を
参照してください。

### 結果サンプル

`uv run python -m harness.report --before naive --after tfidf`
を同梱の30問評価セット(トピックの異なる16文書＋定型的な文言を多く含む
おとりのFAQ文書1件)に対して実行すると、naiveベースラインからTF-IDF
リトリーバーへ小さいながらも一貫した改善が見られます(例: MRR 0.983→
1.000、Hit@1 0.967→1.000)。これはIDF重み付けによって、おとり文書に
繰り返し出現する定型的な語句の影響をTF-IDFが無視できる一方、重み付けの
ないnaiveベースラインはそれに惑わされて誤った文書を上位に出すことが
あるためです。

このn=30という評価セットの規模(各指標で1問分の入れ替わりに相当)では、
並べ替え検定はこの改善を「偶然のばらつきと統計的に区別できない」
(p≈1.0)と正しく報告します。これは意図的に調整せず残しています。
少ないサンプルから誤った確信を作り出さないこと、そして「After実行の
スコアが高かった」というだけでは実質的な改善の証拠にはならないことを、
このハーネス自体が示すためです。実際の数値は実行環境により変動し、
固定シード(`--seed`、既定値`42`)によって再現可能です。README内の数値
を鵜呑みにせず、実際にコマンドを実行して現在値を確認してください。

### テスト

```bash
uv run pytest
```

`metrics.py`・`stats.py`・`runner.py`・`report.py` それぞれについて正常系・
異常系・境界値のテストを備え、同梱の `data/` を実際に読み込むエンドツー
エンドの統合テストも含みます。

### ライセンス

本リポジトリにはMIT等のオープンソースライセンスは付与されていません。
利用条件は本ファイル末尾の記載に従います。

---

© 2026 株式会社日本FAD. All rights reserved. This repository is published for demonstration purposes only. No license is granted for reuse, modification, or redistribution.
