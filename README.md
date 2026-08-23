# Twisty Puzzle AI Lab

Twisty Puzzle AI Lab は、ツイストパズルを対象にした探索・学習実験用の Tkinter GUI アプリケーションです。
Rubik's Cube 系を中心に、複数パズルの状態表示、手動操作、自動ソルブ、学習データ収集、ニューラルネットワーク学習、解析ビューアをまとめて扱います。

現在の `main.py` の既定設定は `cube_size = 7`, `puzzle_type = 'cto'` です。
実験対象や AI 設定は `build_default_frame_config()` から変更します。

## 主な機能

- 手動操作と自動ソルブを同じ GUI で実行
- Search2: Policy / Value を使う priority 型探索
- Search3: PUCT ベースの探索木探索
- myperms: 既知手順・発見手順を使う Greedy fallback
- Search2 / Search3 学習データの登録、簡約、保存、再利用
- Original MLP / Residual / Attention 付き Original Transformer 風モデルの実験
- NumPy 実装と PyTorch 推論・学習の切り替え
- Grad / Integrated Gradients / Occlusion / Attention / Embedding 解析
- Dataset Inspector と sample replay
- 学習ログ、探索ログ、last_perms、lp_show、パラメータ編集 GUI

## 対応パズル

`FrameConfig.puzzle_type` で切り替えます。

- `cube` / `rubiks`: 2x2 から多分割 Rubik's Cube 系
- `megaminx`: Megaminx
- `pyraminx`: Pyraminx
- `master_pyraminx`: Master Pyraminx
- `skewb`: Skewb
- `square1`: Square-1
- `fto`: Face-Turning Octahedron
- `cto`: Corner-Turning Octahedron
- `group`: 対称群または素体上の有限線型群（`group_main.py` から起動）

各パズルの状態遷移は `cube/`, `megaminx/`, `pyraminx/`, `skewb/`, `square1/`, `fto/`, `cto/` に分かれています。
パズル固有の StateViewer は `ui/<puzzle>/state_viewer.py` にあります。

## 実行方法

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 main.py
```

Python 3.10 以降に対応し、Python 3.14 で動作確認しています。Tkinter の GUI が起動します。macOS などで Tkinter が入っていない Python を使っている場合は、Tk 対応の Python を使ってください。

### 有限群パズル

既定の対称群パズル（`S_3`）は次で起動できます。

```bash
python3 group_main.py --kind symmetric --degree 3
```

線型群の既定例（`GL(2, F_2)`）は次です。

```bash
python3 group_main.py --kind linear --dimension 2 --modulus 2
```

対称群の状態は 1-based one-line 表記、線型群の状態は行列の row-major flatten です。生成元は常に左から作用します。AI 入力は、対称群では各位置ごとの `n` 次元 one-hot、線型群では各行列成分ごとの `p` 次元 one-hot を連結したものです。

独自の生成元は `build_group_frame_config(..., generators=...)` へ辞書で渡します。対称群では `1..n` の置換、線型群では `F_p` 上の正則行列を指定してください。生成元集合に各逆元が含まれない場合は起動時にエラーになります。

```python
from group_main import build_group_frame_config

config = build_group_frame_config(
    kind="symmetric",
    degree=3,
    generators={
        "s": [2, 1, 3],
        "r": [2, 3, 1],
        "r^-1": [3, 1, 2],
    },
)
```

`group` mode は myperms を登録・利用せず、level `L` のスクランブルを生成元集合 `X` の長さ `L` のランダム word で作ります。探索器へ渡るのは結果の群要素の状態表現です。
学習パラメータは生成元設定ごとにローカルの `GroupAIdatas/` 以下へ保存され、cube 用の `AIdatas*` とは分離されます。これらの生成データはリポジトリには含めません。

`SL_2(F_7)` の編集可能な実験例は `group_experiment.py` にあります。

```bash
python3 group_experiment.py
```

このファイルの `DIMENSION`, `MODULUS`, `GENERATORS`, AI数、`AI_SEARCH_MODES`, `FRAME_OPTIONS` を変更すれば、実験条件を一か所で差し替えられます。`AUTO_ADD_INVERSES = True` の場合、不足する逆元を自動追加します。既定例の入力 `A, B` は内部で `X = {A, A^-1, B, B^-1}` になります。AI構成は cube 実験と同じく、Search2の通常Linearモデル10体とpiece-token Transformerモデル10体です。

Tools の `embedding map` と `embedding analysis` は群入力にも対応します。線型群では各点を `a00 / value=5` のように表示し、対角・非対角成分または行で色分けできます。通常Linear AIでは `W1`、Transformer AIでは `W1` に加えて `WQ1 @ W1`, `WK1 @ W1`, `WV1 @ W1` を解析できます。

## 依存関係

必須（`requirements.txt`）:

- Python 3.10 以降
- NumPy 1.23 以降、3.0 未満
- heapdict 1.x
- Tkinter

任意:

- PyTorch 2.1 以降: Transformer / Torch 推論・学習を使う場合

例:

```bash
python3 -m pip install -r requirements-torch.txt
```

`torch` は環境依存が大きいので、不要なら入れなくても NumPy 側の処理は動きます。

## テスト

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests -v
```

GitHub Actions では Python 3.10 と3.14の両方で同じテストを実行します。

## 設定の変更

基本設定は `main.py` の `build_default_frame_config()` で行います。

よく変更する項目:

- `cube_size`: Rubik's Cube 系のサイズ
- `puzzle_type`: 起動するパズル種別
- `ai_search_modes`: 各 AI の探索方式 (`search2`, `search3`, `transformer`)
- `search3_progress`: Search3 budget 時に fallback prefix へ進めるか
- `search3_cs`: Search3 PUCT の C
- `lrs`, `wdlrs`, `adam`, `weight_decay`: optimizer 設定
- `update_scales`: shared / policy / value head の更新スケール
- `residuals`: Original AI の residual block 有効化
- `original_transformer_attention`: Original AI に attention block を入れるか
- `transform_idx`, `flip_inside_idx`: 変換・鏡映を固定するか
- `priority_list`: Greedy fallback で注目する group の順序
- `bootstrap_datas`, `bootstrap_search3_datas`: 起動時に追加する初期データ

GUI 起動後は Tools から parameter editor / dataset inspector / attention analysis などを開けます。

## 探索と学習データ

### Search2

Search2 は、Policy 出力で候補手を広げ、Value が高い状態を優先する探索です。
探索結果の move sequence は簡約して Search2 データへ登録します。

### Search3

Search3 は PUCT 型探索です。
各ノードは Policy prior と Value backup を使って選択されます。
Search3 の学習データは solve 履歴から作られ、move sequence は簡約後に登録されます。
Value target は残り手数と `value_target_gamma` を使って再計算されます。

Search3 データには以下の区別があります。

- `search_mode`: データ片を生成した探索方式
- `end_reason`: `solved`, `budget`, `budget-greedy`, `bootstrap` など
- `source_succeeded`: その探索結果自体が成功したか
- `solve_succeeded`: その solve 全体が成功したか

### myperms / Greedy fallback

`cube.rubiks_cube.Rubiks_3.myperms` には、交換・回転・flip・parity 補正などの手順が登録されています。
Search が解を見つけない場合でも、myperms を評価して Greedy に進めることがあります。
発見した短い手順は `last_perms` としてログ・表示されます。

全パズルの `myperms` key は `MypermKey(name, transform_index)` で統一されています。
登録手順の `name` は代表変換の効果名を使い、画面・ログでは `name#NN` と表示します。
旧手順名はaliasとして解決できます。
単一 move は `SingleMove-<move>` という名前を使います。
効果ベースの命名規則と向きの表現は [`docs/myperm_naming.md`](docs/myperm_naming.md) にまとめています。

## 解析ツール

GUI の Tools から主に以下を使います。

- `edit params`: AI パラメータの確認・編集
- `dataset inspector`: Search2 / Search3 データセットの概要、代表 sample 表示、sample replay
- `attention analysis`: Attention の重要度・関係性の表示
- `embedding analysis`: piece embedding の norm / PCA / cosine 類似度表示
- `lp show`: last_perms の手順表示
- `make myperm`: 現在の手順から myperm 登録用 key を作る補助

GradViewer 系では以下の mode を切り替えられます。

- `Grad`
- `IG`
- `Occ`
- `PieceOcc`
- `PolicyOcc`
- `PiecePolicyOcc`
- `AttnIn`, `AttnOut`, `AttnCentral`
- `EmbNorm`, `EmbPC1`

## ディレクトリ構成

```text
main.py                  起動設定とアプリ起動
group_main.py            有限群パズルの起動入口
group_experiment.py      有限群の編集可能な実験例
ai/                      Original AI、layer、loss
core/                    共通定数、scramble selector
cube/                    Rubik's Cube 系、Search2/Search3 engine、myperm analyzer
megaminx/                Megaminx の状態・操作
pyraminx/                Pyraminx / Master Pyraminx
skewb/                   Skewb
square1/                 Square-1
fto/                     FTO
cto/                     CTO
group_puzzle/            対称群・有限線型群パズル
managers/                solve / learn / params / debug / dataset 管理
model/                   SearchResult と学習データ構造
ui/                      Tkinter UI、viewer、dialog
tools/                   独立実行用の補助ツール
tests/                   自動テスト
docs/                    設計・命名規則の詳細
reports/                 ツールが生成する解析レポートの説明
```

## データとモデル

学習済みパラメータ、optimizer state、収集した学習データ、生成レポートは実行時の成果物として扱い、Git では管理しません。

- `AIdatas*/`: 各 AI のパラメータ
- `GroupAIdatas/`: 有限群実験のパラメータ
- `Datas/`, `Datas.binaryfile`: 収集した学習データ
- `reports/*.csv`: 補助ツールが生成する解析レポート

パラメータファイルが存在しない場合、モデルは初期値から起動します。公開用の学習済みモデルを配布する場合は、GitHub Releases などから個別にダウンロードする形を想定しています。

## Git 管理上の注意

仮想環境、キャッシュ、OS・エディタ生成ファイル、学習データ、モデルパラメータ、生成レポートは Git 管理しません。特に `.npy` を大量に含む `AIdatas*/` などは、リポジトリサイズを急増させるためコミットしないでください。

## 開発メモ

このリポジトリは研究・実験用のコードです。
パズル種別、探索方式、学習方式、解析機能が同時に入っているため、動作確認時は `main.py` の `FrameConfig` と現在の `puzzle_type` を必ず確認してください。
