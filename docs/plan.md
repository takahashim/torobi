# Torobi(とろ火)実装計画 v3.2

- Status: Proposal(v3.1 への外部レビューを受けて改訂。M2 着手の前提)
- Date: 2026-09-03
- Primary target: Apple Silicon macOS
- 名称: gem / リポジトリ = `torobi`、エンジン crate = `torobi-engine`、namespace = `Torobi::`
- v3.2 の変更(2026-09-03、外部レビュー反映): 実行契約を新設(§5A: batch 経路、
  状態遷移、model/objective 接続、エラー分類)、native 境界の見積もりを訂正(§4.1、
  実測で反証された)、replay を 2 モードに分離(§8.6)、マイルストーンを G0/M1〜M4 に
  再分割(§9.1)、v1 文書への規範参照を本文に取り込み(§11、§12)、配布 feasibility を
  M1 に前倒し、計測主張を仕様化
- 再検討反映(2026-09-03): burn-rb は参考に格下げ、プロセス同居禁止の撤回を明文化、
  ModernBERT を「最初のとっかかり」に一般化、製品面を 3 層構造に再編、配布方針を未確定に変更
- v3 からの主な変更: エンジンを in-process(狭い腰の native 拡張)に転換、
  GraphConfig を model / objective の 2 部構成に、介入モデルを「窓の契約」として定式化、
  歴史的主張を弱め設計根拠を限定スコープに置き直し

## 0. 性格(なぜ「とろ火」か)

- 納期優先で学習させたいなら、クラウドの CUDA GPU を回すべきである。Torobi はその道具ではない。
  手元の Mac で、時間をかけて、宣言的に、安全に煮る。速さを競わない。
- 作者が頑張りたいのは Graph DSL / IR の設計と数値の検証であり、広い FFI 面の保守や
  学習ループの所有ではない。GVL / GC への対処は「面を狭めて一度だけ払う」(§4)。
- **Ruby が言語を所有し、Rust は実行だけを所有する。逆流したらそれは設計の後退である。**

### 0.1 設計根拠(意図的に弱い主張にする)

「Transformer 時代だから静的グラフが正しい」とは主張しない。主張するのは:

> Torobi が対象とする既知アーキテクチャの FT / 蒸留 / LoRA では、モデル本体の
> トポロジーを静的 IR として扱うコストが十分小さくなった。

- define-by-run が主流になったのは、動的アーキテクチャ(RNN / tree)への必要だけでなく、
  「Python で普通に書け、デバッガで追え、途中の tensor を見られ、即試せる」体験による。
  この便利さは Transformer が静的になっても消えていない。Torobi はこのうち
  「途中の tensor を見る」を観測タップ(§8)で、「即試す」を Python ランナー(§10)と
  窓(§8)で回収し、残りは対象領域の限定によって影響を小さくする。
- コンパイラ / 静的実行系の隆盛は「Transformer が静的だから」ではなく、計算量の巨大化で
  fusion・memory traffic・kernel specialization の経済価値が跳ね上がったことによる。
  torch.compile は eager から静的領域を安全に回収する技術(graph break、dynamic shapes 込み)
  であり、「静的への回帰」ではない。
- Torobi は define-and-run 側に立つが、これは PyTorch と競う選択ではなく、
  ONNX / MIL と同じ層(交換・実行 IR)をフロントエンドごと所有する選択である。
- **数値の主張は計測仕様とセットでしか書かない。** 「境界コスト 0.1% 未満」「cosine ≈ 1.0」
  「一晩以内」の類は、hardware / shape / dtype / dataset / 許容値を明示した出口条件
  (§9)に置き換える。裸の比率や倍率を設計根拠にしない。
- アーキテクチャ差分が config に縮退した、とも主張しない。Torobi の扱う範囲では
  そう見なせるモデルが十分多い、に留める。IR の語彙は
  「config パラメータ + 少数の block variant + named semantic ops」で構成する(§6)。

## 1. Summary

burn-rb の Graph DSL を参考にしつつ(書き味の互換は目的にしない)、実行系に MLX を使う
Ruby の学習基盤。Ruby はモデルと目的関数を宣言的に記述し(構築時に一度だけ実行)、
学習の実行は in-process の Rust エンジンが担う。Ruby と計算の境界は「狭い腰」の
セッション API から始める(§4。狭さは姿勢であって教義ではない)。

```text
torobi gem
  ├ Ruby 層(このプロジェクトの本体)
  │   ├ Graph DSL → GraphConfig(model graph + objective graph、JSON、digest)
  │   ├ Experiment クラス、セッションプログラム、フック、実験台帳、weights surgery
  │   └ journal / replay
  └ native 拡張(狭い腰: セッション API 十数関数、magnus)
        │  callback なし・closure なし・テンソル非越境(コピーのみ)
        ▼
   torobi-engine(Rust crate)
        ├ GraphConfig を serde で解釈、step 実行・optimizer・checkpoint を所有
        └ vendored mlx-rs → mlx-sys → mlx-c → MLX / Metal
```

設計の中心:

1. 数値計算と autodiff は MLX に任せ、Ruby でも Rust でも再実装しない。
2. Ruby の自由度は「構築時」と「窓」(§8)に限定し、step の内部を純粋関数に保つ。
3. **決定は自由。ただし決定は宣言された経路を通る**(§8.1)。
4. native 境界の危険は面の広さに比例する。腰を狭く保つことが GVL / GC 対策の本体(§4)。

## 2. v1 / v3 からの主要変更

| 項目 | v1(mlx-graph-rb) | v3 | v3.1(本書) |
|---|---|---|---|
| native 層 | 広い C 拡張(op / Array / closure) | 別プロセス + JSONL | **in-process の狭い腰**(セッション API のみ) |
| GVL / GC | 設計の中心 | 消滅 | **有界**: 呼び出し点数箇所、callback ゼロ、テンソル非越境 |
| リモート実行 | なし | 可能(IPC) | **不要と判断し削除** |
| GraphConfig | model のみ | model のみ | **model + objective の 2 部**(§6) |
| lowering | 全て primitive へ | fast op は例外 | **semantic op 保持 + backend decomposition fallback**(§6.1) |
| 介入 | hook 限定 | stop/resume 合成 | **窓の契約**: 粒度自由・能力列挙・フックは糖衣(§8) |
| 事前学習済み重み | 対象外 | 一級市民 | 同左 |
| oracle | Python MLX | + kohagi | 同左 + **評価 3 軸**(§9.2) |

v1(mlx-graph-rb 案)の GraphConfig 設計と op registry manifest の考え方は §6 に、
乱数・checkpoint・correctness の規範は §11・§12 に**本文として取り込んだ**。v1 文書は
`notes/`(git 管理外)にしか無く、規範をそこへ参照で預けることはしない。

## 3. Goals / Non-goals

### Goals

- 事前学習済み checkpoint(HF safetensors)を初期値として FT / 蒸留 / LoRA ができる。
- 既知アーキテクチャを Graph DSL で記述できる。最初のとっかかりは ModernBERT
  (local/global 交互注意、二重 rope theta、GeGLU)で、以後 decoder 系(Llama 型)等を
  順次追加する。語彙はアーキテクチャ非依存に設計し、特定モデルへの特化を避ける。
- 実務家の面(実験定義・データ・台帳・評価・介入)が 100% Ruby である。
- 学習の観測と介入が「窓の契約」(§8)の範囲で足りる。

### Non-goals

- step 内部での Ruby コード実行(数式・微分経路への介入)。行き先は §8.4 / §10
- in-process の広い eager Array API、custom VJP、Metal カーネル注入
- リモートエンジン、分散
- スループット競争、および CUDA / Linux 対応。MLX 自体には CUDA と Linux CPU の backend が
  あるので「使えないから」ではなく、**製品スコープ上の選択**として当面対象外にする
  (対象は手元の Mac での FT / 蒸留)
- 以下は引き続き対象外: Python MLX API 互換、Python dunder の再現、学習中の任意 Ruby
  forward、DLPack、custom Metal kernel、任意 Ruby オブジェクトを含む parameter tree、
  MLX の全 dtype / 全 op / 全 optimizer の公開、`Marshal` による保存、初期段階での
  training step 全体の compile

## 4. native 境界: 狭い腰のセッション API

エンジンは magnus で束ねた in-process ライブラリとし、Ruby から見える native 面を
**十数個の関数**に限定する。

- セッション生命(open / close)、`run_steps(n)`、ノブ調整、metrics 取得、
  checkpoint / rollback、fetch / put(名前指定・コピー)、タップ設定、eval 実行
- **Ruby の Proc / closure を native に渡さない**。テンソルはハンドルでなくコピーで越境する
  (大半はスカラー / JSON。fetch は明示操作)
- GVL は長い計算を包む数箇所のラッパでのみ解放する。スパン(§8.2)は
  「step 粒度の native 呼び出しを Ruby 側でループする」形で実装し、GVL 非保持時間を
  1 step 分に有界化する(フックの自動分割と同一機構。呼び出し境界のコストは
  step 実行時間の 0.1% 未満)
- native 側に Ruby が保持する長寿命オブジェクトはセッション 1 個。GC 連携は
  TypedData の free のみで、mark / compact を要する型を作らない

### 4.1 境界は狭いが、閉じてはいない(v3.1 の見積もりの訂正)

v3.1 は「native 境界は一度書いて閉じる有界な問題」と書いた。**これは楽観であり、
実測で反証された**。現 HEAD で確認した事実:

| 失敗 | 実際の挙動 |
| --- | --- |
| MLX が C エラーハンドラ経由で報告する失敗(shape 不一致など) | Ruby の例外になる。`rescue` できる |
| デバイス / ライブラリ初期化の失敗(metallib 不在) | **プロセスが exit 255 で死ぬ**。C++ 例外は Rust が捕捉できず、Ruby に届かない |
| Rust の panic | GVL トランポリンが `catch_unwind` し、GVL 復帰後に resume。C を貫通しない |

したがって境界の性質は「狭い」であって「安全」ではない。方針:

- 予見できる致命は **MLX に触れる前に Ruby 側で拒否する**(`Torobi::Preflight`)。
- 予見できないものは残る。**subprocess で異常系を回すテスト**を常設し、abort する事実
  そのものをテストで固定する(挙動が黙って変わらないように)。
- ASan / UBSan は Rust・拡張の通常テストとして CI に置く。

境界の出口条件は §9.1 の M1 に列挙する。

なお、プロセス分離は要件ではない(v3 の「同居させない」原則は撤回済み)。腰の狭さも
教義ではなく費用対効果の姿勢であり、価値が実証されれば面を広げる選択(例: 読み取り専用の
限定的な eager API)を妨げない。ただし広げる際は §12「狭い腰の劣化」の審査
(テンソル越境と callback の禁止を破らないか、関数数の増分に見合う価値か)を通す。

## 5. 依存(v3.2 で方針変更: vendoring ではなく pinned git)

- torobi-engine は **OminiX の mlx-rs を rev 固定の git 依存**として使う。
  v3.1 までは「選択的 vendoring(array / ops / fast / transforms / io を取り込み、
  残りを刈る)」としていたが、**同じ目的をより安く達成できるため変更した**:
  自分のツリーに 2.3 MB を抱えず、Cargo.lock が commit を強制し、どこででもビルドできる。
  失うのは刈り込みとローカルパッチで、必要になったら fork に pin し直す。
  経緯と実測は `docs/vendoring.md`。
- nn / optimizer はエンジン内で所有(mlx-rs は参考)。AdamW は oracle の数 step と照合する。
- 依存台帳(rev、MLX / mlx-c の出所、metallib の配置制約)を維持し、
  `Torobi::Native.build_info` が報告する。更新は一版ずつの明示作業。

## 5A. 実行契約(G0。M2 着手の前提)

v3.1 は責務分離を書いたが、**実行の契約を書いていなかった**。以下を IR とセッション API の
仕様として先に確定する。ここが決まるまで M2 には入らない。

### 5A.1 状態遷移としての step

```text
(state', outputs) = step(state, batch, knobs)

state = parameters + optimizer slots + RNG state + counters(step / epoch)
        + sampler position
batch = 名前つきテンソルの束(§5A.2 の経路で供給される)
knobs = lr、損失重み、freeze 集合、…(§8.3 の A)
```

不変条件(§8.1)はこの形で読む: step は state と batch と knobs の関数であり、
Ruby は state を直接書き換えない(ノブと §8.3 B の put を通す)。

**freeze / unfreeze はスカラーノブではない。** 変更すると (1) autodiff の対象集合が変わり、
(2) optimizer slot の保持 / 初期化 / 破棄が要り、(3) compile cache key が変わる。よって
freeze はスパン境界でのみ有効になる**構造変更**として扱い、journal には「どの集合が
いつ変わったか」を記録する。tap 集合の変更、入力 shape / dtype の変更も同じ理由で
再コンパイル要因であり、同じ扱いにする。

### 5A.2 batch の供給経路

Ruby がデータを所有し、かつ計算中に Ruby コードを呼ばない、を両立させる経路を選ぶ。
候補と評価:

| 案 | 形 | 評価 |
| --- | --- | --- |
| (a) per-step 供給 | `run_step(batch)` を Ruby が毎 step 呼ぶ | span = 1 に固定される。境界コストは実測次第(§9.1 M1 の出口条件) |
| (b) 事前投入キュー | Ruby が N batch を native キューへコピー → `run_steps(n)` が消費 | 長いスパンが可能。backpressure と prefetch の設計が要る |
| (c) native BatchSource | データ供給を engine に持たせる(ファイル読み) | Ruby がデータを所有する目標に反する。採らない |

**encoding は typed である。** batch は `{name => [dtype, shape, packed]}` で渡り、
payload は native-endian の 4 byte 値。dtype が渡るのは、graph が i32 入力を宣言できる
(embedding が読むもの)以上、f32 を仮定した境界ではそれを運べないため。

**方針: (a) を基本、(b) を最適化として後から足す。** (a) は契約が単純で、Ruby のデータ層
(ActiveRecord / pgvector / JSONL)がそのまま活き、可変長 batch も自然に扱える。(b) は
実測で (a) の境界コストが問題になったときにだけ導入する。どちらでも
**batch はコピーで渡る**(ハンドルなし)ことは変えない。

決めるべき細部(G0 の成果物):
- 可変長 batch の表現(shape はリクエストごとに変わる。§6.2 の static topology ≠ static shape)
- コピーのコストと、それが step 時間に占める割合(計測して記録する)
- キャンセル(§8 の窓が開く前に止めたい場合の意味論)

### 5A.2.1 計測(2026-09-03、M2 Mac、`bench/boundary.rb`、dim=128、50 steps)

| rows | marshal | step | span | 差 |
| --- | --- | --- | --- | --- |
| 1 | 0.010 ms | 0.400 ms | 0.445 ms | 誤差 |
| 8 | 0.069 ms | 0.536 ms | 0.539 ms | 誤差 |
| 64 | 0.534 ms | 1.179 ms | 1.213 ms | 誤差 |
| 512 | 4.280 ms | 6.629 ms | 6.876 ms | 誤差 |

読み取れること、そして**この計測が上の分析を一部覆した**:

1. **per-step 呼び出しと span はコストが変わらない**。呼び出し境界そのものは計測の
   ノイズに埋もれる。よって (a) を基本とする判断は裏付けられ、**(b) 投入キューは
   「呼び出しコストを減らす」という理由では正当化されない**。
2. **境界の実体は JSON 直列化である**。512 行では step 6.63 ms のうち marshal が
   4.28 ms(65%)。batch が大きいほど支配的になる。
3. したがって最適化するなら投入キューではなく **encoding**(packed binary)である。
   G0 の作業項目をそちらへ移した。

**packed encoding 後**(同条件。shape は JSON のまま、payload は native-endian f32):

| rows | json(旧) | pack(新) | step | span |
| --- | --- | --- | --- | --- |
| 1 | 0.009 ms | 0.003 ms | 0.382 ms | 0.466 ms |
| 8 | 0.071 ms | 0.013 ms | 0.441 ms | 0.425 ms |
| 64 | 0.535 ms | 0.095 ms | 0.530 ms | 0.485 ms |
| 512 | 4.320 ms | 0.730 ms | **1.252 ms**(旧 6.629) | 1.286 ms |

512 行の step が 5.3 倍速くなった。encoding は依然として大きな batch では step の
半分強を占めるので、次に効くのは「呼び出し側が最初から packed で持つ」ことである
(`Torobi::Batch.pack` は String をそのまま通す)。**投入キューは引き続き不要**。

### 5A.3 model graph と objective graph の接続

以下を GraphConfig の正式な型として定義する(v3.1 は「入力に取る」としか書いていなかった)。

- **`ModelOutputRef`**: objective の入力として `{"model": "student", "output": "logits"}`
  を指す参照型。model graph の出力に**名前**を要求する(現状の位置指定 outputs を
  名前つきに拡張する)。
- **`BatchRef`**: batch のフィールドを指す参照型 `{"batch": "teacher_logits"}`。
- **model output の契約**: 名前・shape・dtype を model graph 側が宣言し、objective の
  shape 推論はそれを使う。不一致は構築時に拒否。
- **`stop_gradient`**: op として IR に持ち、**明示的に呼ぶ**(`g.stop_gradient(x)`)。
  v3.1 は「teacher の出力は既定で通す」と書いていたが、暗黙の挿入は「どこで勾配が
  止まったか」を graph から読めなくするので採らない。凍結 model のパラメータは
  そもそも argnums に入らないため、既定にしなくても teacher は学習されない。
- **parameter ownership**: パラメータは model graph に属し、path は
  `student.layers.0.wqkv.weight` のように **model 名で名前空間化**する。
- **autodiff の対象**: `trainable == true` かつ freeze されていないパラメータのうち、
  **学習対象と宣言された model のもの**だけを argnums に含める。teacher のパラメータは
  含めない。この集合は GraphConfig と knobs から決定的に定まる。
- **weight sharing と複数回呼び出し**: 同一 model を objective 内で複数回参照できる。
  同じ parameter id を参照するので共有は IR レベルで自然に成立する。オンライン teacher の
  重複実行は、同じ `ModelOutputRef` を複数箇所で使った場合に **1 回に畳む**(共通部分式の
  除去)ことを engine 側の契約とする。

### 5A.4 エラーの分類

§4.1 の実測に基づき、3 分類を仕様とする。

1. **構築時エラー**(`ConfigError`): 構造・shape・dtype・所有権・objective 契約(§5A.3)。
   Ruby で完結し、native に触れない。
2. **実行時エラー**(`StepError`): MLX が C エラーハンドラ経由で報告するもの、および
   engine が拒否したもの。例外に変換され、**セッションは生存する**(次の step を試せる)。
   これを保証するため step は transaction である: 次の parameters / slots / RNG を作って
   eval し、成功して初めて session がそれになる。
3. **致命**(`EngineUnavailable` で予防、あるいはプロセス終了): 初期化・デバイス・
   ライブラリ不在。予見できるものは preflight で拒否し、できないものは
   **プロセスが死ぬことを仕様として認める**(supervisor 前提。§9.1 M1 の異常系テスト)。

## 6. GraphConfig: model graph + objective graph の 2 部構成

動性はモデルのアーキテクチャから **training program** へ移った(外部レビューの中心的指摘)。
これを受け、GraphConfig を 2 部構成にする。

- **model graph(s)**: encoder / head。複数持てる(teacher / student)。
- **objective graph**: batch のフィールドと各モデルの出力を入力に取り、スカラー損失に至る
  微分可能な計算。蒸留の teacher/student 配線、損失合成(margin-MSE + InfoNCE + Matryoshka)、
  masking、sample weighting、multi-task の重み付き合算はここに宣言的に載る。
  しきい値や重みは「入力」にでき、窓のノブ(§8)で動かせる。
  オフライン教師(事前計算 logit)もオンライン教師(同 step で forward)も同じ形式。

### 6.1 semantic op と decomposition(v3 の「lowering の例外」を置換)

PrimTorch 型の層構造を採る。IR は semantic op(sdpa / rope / layer_norm / rms_norm / geglu、
将来 varlen_attention 等)を**保持**し、実行時に backend が選ぶ:

```text
semantic op ── backend が fused kernel を持つ → それを使う(MLX fast 系)
           └─ 持たない → primitive へ decomposition(fallback)
```

「絶対に lowering しない」原則ではない。decomposition は常に定義され、
fused 実装との一致は §9.2 の 3 軸で検証する。

### 6.2 static topology ≠ static shape

グラフの**トポロジー**は静的でも、**データ**は動的でよい。varlen(cu_seqlens)、
topk / gather(MoE ルーティング)、データ依存 masking は、制御流ではなく op として
静的グラフに載る。IR はこの区別を前提に設計する。

### 6.3 事前学習済み重み

ParameterSpec の initializer に `{"type": "pretrained", "source": …, "tensor": …}` を追加。
HF 名との写像は GraphConfig に記録され、shape / dtype 不一致は load 時に fail closed。
`trainable: false` と組み合わせて freeze を表現し、LoRA はその上の定型とする。

### 6.3A 接続の型

§5A.3 の `ModelOutputRef` / `BatchRef` / `stop_gradient` / 名前つき model output /
model 名による parameter 名前空間は、GraphConfig の型として定義する(G0 の成果物)。

### 6.4 ノードの自動命名

DSL は全モジュール境界・全中間ノードに安定した path 名を自動付与する
(ParameterSpec の path と同じ規律)。観測タップ(§8.3)と freeze パターンの前提。

## 7. Ruby 製品面(3 層)

v3 の §7 は Rust CLI 時代の設計(Experiment class macro を最上位に置く)を引き継いでいたが、
Torobi の本体が Graph DSL である以上、製品面は層で示す。

### 7.1 核: Graph DSL(モデルと目的関数の記述)

Torobi の本体。モデル(§6 の model graph)と目的関数(objective graph)を Ruby で宣言する。
書き味は M0 で設計する(burn-rb は参考に留め、互換は目的にしない)。方向性の例示:

```ruby
reranker = Torobi.graph do |g|
  ids = g.input :ids, [nil, nil]                    # [batch, seq]
  h   = g.embedding(vocab:, dim:)[ids]
  cfg.layers.times { |i| h = modernbert_block(g, h, i, cfg) }  # 部品化はただの Ruby メソッド
  g.output g.classifier(h[:, 0, :], labels: 1)
end

objective = Torobi.objective(student: reranker) do |g, batch|
  g.output margin_mse(g.student.logits, batch[:teacher_logits])
end
```

部品化・条件分岐・繰り返しは通常の Ruby(構築時に一度だけ実行)。生成物は 2 部の
GraphConfig で、digest 付きの成果物になる。

### 7.2 実行: セッションと窓

§8 の契約に従う実行 API(`Torobi.session`、`s.run` / ノブ / フック / journal)。

### 7.3 応用層: 実験の定型(形は後で決める)

FT / 蒸留の定型(student / teacher / freeze を束ねる Experiment 風の器、実験台帳、sweep)は
応用層であり、M4 のドッグフーディングを終えてから形を確定する。class macro 基盤を
先回りして作り込まない。それまでの実験は 7.1 + 7.2 の素の組み合わせ(ただの Ruby
スクリプト)で書く。

weights surgery(部分ロード、remap、LoRA merge、checkpoint 補間)と journal / replay は
応用層ではなく核付属のユーティリティとして提供する。

## 8. 窓の契約(v3 の「closed loop の契約」を置換)

### 8.1 不変条件

> **1 step の内部は、IR が定義した純粋関数である。Ruby の影響は宣言された経路
> (入力・パラメータ・ノブ)だけを通り、効力は step 境界(構造的な変更はスパン境界)で
> 発生する。決定は自由。ただし決定は宣言された経路を通り、journal に残る。**

step 内のデータ依存の決定は op として IR に(§6.2)、step / スパン境界の決定はノブとして
Ruby に住む。除外されるのは「経路を通らない実行」(Ruby コードが計算の内側で走ること)のみ。

### 8.2 粒度: 自由

- スパン = `s.run` 1 回に対応する連続実行区間。**約束ではなく介入頻度の選択**であり、
  1(毎 step)から :all(全自動)まで任意。`s.each_step` は span=1 の糖衣。
- フックの `every:` 宣言はスパンを自動分割する。粒度指定は実質フックに一本化できる。
- step 境界の呼び出しコストは実行時間に対し 0.1% 未満(§4)。粒度は性能の議論ではない。

### 8.3 能力: 列挙(A / B / B+ / C)

| 段階 | 内容 | 扱い |
|---|---|---|
| A | metrics・損失項別値・grad/param norm を読む。名前付きノブを回す(lr、損失重み、freeze/unfreeze、データ切替、停止、checkpoint、rollback、eval) | どの粒度でも可。journal 記録 |
| B | 重み・optimizer 状態・勾配の名前指定 fetch / put(**コピーのみ**) | 可。put は journal 記録 |
| B+ | **観測タップ**: 任意の名前付きノードを読み取り専用で step の出力に昇格。`stats:`(mean / norm / histogram)でエンジン側縮約可 | 可。読み取り専用のため journal 不要。fusion / メモリへの影響に注意(常設は stats 縮約を推奨、タップ集合の変更は再コンパイルを伴う) |
| C | step の数式・微分経路への Ruby の介入、live テンソルの直接改変 | 除外。行き先: (1) objective graph の語彙拡張(しきい値ノブ付き op で「選別的 backprop」等も宣言できる)、(2) span=1 の 1 step 遅れ制御、(3) §10 の Python ランナー。「forward を見て同 step の backward を選ぶ」形は、必要が実証されたら peeked step として名前付き機能で検討(二重計算コストを明示) |

### 8.4 フック: 窓の糖衣

- `s.on(event) { }` と `s.use(policy)` の 2 口。発火点は列挙(steps / plateau / nan /
  eval_done / checkpoint_written / span_end)。**フックは窓でのみ発火し、能力は A / B / B+ と
  完全に同一**(新しい危険を持ち込まない)。
- 柵: 発火順は登録順で決定的。フックから `s.run` は呼べない(再入禁止)。フック内の例外は
  窓で run を停止する(窓では状態が常に一貫)。フックの調整も同じノブ API を通り journal に残る。
- 標準ポリシー: NaNGuard(rollback)、BestCheckpoint、LrOnPlateau、EarlyStopping、Progress。
- 規範: **実験の主筋はセッションプログラム(上から読める)、横断的関心はフック**。

### 8.5 journal

窓での操作(ノブ、put、データ切替、freeze の変更)は名前付きで journal に自動記録される。
安定したセッションプログラムは recipe(宣言)へ昇格できる。

**記録対象は操作だけではない。** 再現に必要なものを列挙する(v1 §13 を本文化した §11 と対):

- GraphConfig の digest と semantics version
- dataset / sampler の状態と、データ成果物の digest
- optimizer の種別と設定、RNG state、step / epoch / batch position
- Ruby 側 policy / hook のコード版(gem version と、ユーザーコードの digest)
- runtime / build 情報(gem、Ruby、MLX、mlx-c、mlx-rs の revision、macOS / arch)
- dtype、backend、compile signature
- **窓で読んだ値**(metrics、tap)。v3.1 は「読み取りは journal 不要」としたが、
  **読んだ値から Ruby が判断する以上、判断の入力は記録が要る**。訂正する。

### 8.6 replay の 2 モード(v3.1 の「決定的に再演」の精密化)

「同一 seed から決定的に再演」は曖昧であり、2 つの別の目的を混ぜていた。分離する。

| モード | policy | 照合対象 | 用途 |
| --- | --- | --- | --- |
| **action replay** | 再実行しない。journal の操作をそのまま適用 | 最終 state と metrics | 学習の再生産、CI |
| **deterministic rerun** | 再実行する(同じ Ruby コード) | 観測値と判断まで含めて照合 | policy の回帰検出 |

一致条件も 3 段に分ける(§9.2 の 3 軸と同じ精神):

1. **bitwise**: 同一マシン・同一 build・同一 dtype で期待する。checkpoint の round-trip など
2. **dtype 別 tolerance**: 演算順序が変わりうる経路(fused / decomposition、並び替え)
3. **統計的一致**: 最終 metric の一致(学習全体の再現。乱数と非決定性を含む場合)

どの一致条件を要求するかは、テストごとに明示する。
## 9. 北極星と検証

### 9.1 マイルストーン(v3.2 で再分割)

v3.1 の M2 / M3 は大きすぎ、M3 の出口条件(logit 一致)では backward も optimizer も
memory も検証できないまま M4 の蒸留に入る構造だった。分割し直す。

| | 内容 | 出口条件 |
| --- | --- | --- |
| M0 ✅ | 純 Ruby の IR と Graph DSL | 同一定義 → 同一 digest、構造検証、shape エラーの構築時報告。native 不要 |
| G0 ✅ | 実行契約(§5A) | batch 経路の決定、`ModelOutputRef` / `BatchRef` / `stop_gradient` / 名前つき output / parameter 名前空間 / argnums 規則を型として実装、エラー 3 分類の実装 |
| M1 ✅ | single-step とその境界 | (1) **異なる batch** で forward / grad / update が回る。(2) FFI 異常系: MLX の報告エラーが例外になる、Rust panic が C を貫通しない、致命は preflight で拒否されるか死ぬことをテストで固定、close / 二重 close / 部分初期化失敗。(3) GVL 解放中に別スレッドが進む。(4) **境界コストの実測**(batch サイズ × 系列長ごと。比率の主張はこの数値に置き換える)。(5) **配布 smoke**: 隔離環境へ gem install → require → 1 step |
| M2 ✅ | stateful core | AdamW(oracle の最初の数 step と一致)、RNG state、checkpoint / resume(resume = 連続実行) |
| M2.5 ✅ | 窓 | ノブ、フック、journal、**2 種の replay**(§8.6)、freeze の構造変更としての扱い |
| M3a ✅ | model import | safetensors ロード、**1 ブロックの forward と gradient の parity** |
| M3b ✅ | ModernBERT | 全体の forward / gradient parity、tiny dataset の過学習、memory 予算の実測 |
| **M4** | **蒸留実験** | 固定 dataset / metric / seed での実験が**完走し、記録が残る**こと |

**M4 の出口条件を訂正した。** v3.1 は「素の base-v2 と同等以上」としていたが、それは
データとハイパーパラメータに依存する**研究成果**であり、フレームワークの完成条件ではない。
Torobi の出口条件は「実験が再現可能な形で回ること」であり、蒸留が成功するかは別の問い。

M6(decoder 系 + LLM LoRA、量子化 op、varlen)は M3b 以降の将来として据え置く。

### 9.2 数値評価の 3 軸

fused kernel と decomposition、実装間の比較は次を**別軸で**評価する(「速くて正しい」を
一括りにしない):

1. **semantic equivalence**: 数学的に同じ関数か(FlashAttention 系は数学的意味としては
   同じ関数であり、数値の一致は 2 の tolerance 評価に委ねる。「exact だから一致する」とは
   書かない)
2. **numerical tolerance**: 演算順序由来の差を dtype ごとの許容誤差で判定
3. **performance**: モデル・系列長・ハードごとに実測(一般化した倍率を主張しない)

oracle は Python MLX(fail closed、版付き成果物)と kohagi(encoder 系)。
検証は §12 の層構成による。

## 10. Python MLX の二役(oracle かつ遊び場)

GraphConfig を Python MLX で実行するランナーを、検証専用でなく**公式の脱出口**とする。
define-by-run の自由(任意 forward、custom VJP、カーネル実験)が必要な瞬間はここで実験し、
収束したらエンジンの機能・DSL の語彙へ戻す。GraphConfig が両世界の共通言語。

## 11. 状態、checkpoint、配布

v1 文書(`notes/`、git 管理外)にあった規範をここに取り込む。参照ではなく本文とする。

### 11.1 再現に必要な状態

明示的に管理する: parameter 初期化 seed、dataset shuffle の seed / state、MLX の RNG
state、dropout の state、global step、epoch、batch position(または sampler state)。
compile を導入するときは model / optimizer / random state を compile の入力・出力に
含める(**RNG を hidden capture にしない**)。

同じ seed と同じ checkpoint から再開した実行が、parameter・optimizer state・次 batch の
順序まで連続実行と一致することをテストする(§9.1 M2)。

### 11.2 checkpoint の形式

`Marshal` は使わない。

```text
checkpoint/000003/
├── manifest.json
├── graph.json
├── parameters.safetensors
├── optimizer.safetensors
└── random.safetensors
```

`manifest.json` に持つもの: checkpoint schema version、GraphConfig digest、semantics
version、gem / Ruby / MLX / mlx-c / mlx-rs の version と revision、macOS と arch、
step / epoch / batch position、optimizer の種別と設定、dataset sampler の metadata、
parameter の path・shape・dtype の inventory。

書き込みは一時ディレクトリに行い、flush と検証の後に atomic rename する。読み込み時は
digest・path・shape・dtype・optimizer 種別を検証し、**不一致を silent に無視しない**。

`graph.json` は digest の重複ではない。**digest は description を名指すだけで、復元は
しない**。書いた run が失われた checkpoint を後から読めることが、run checkpoint と
engine-state checkpoint の違いである。読み戻し時に graph.json の digest が manifest の
主張と一致することを確かめる。

**engine が知らないものは engine が書かない。** epoch、batch position、sampler state、
dataset の同一性は、データを所有する側のものである(§5A.2: engine は batch を渡される
のであって取りに行かない)。これらは `run` として JSON のまま書かれ、そのまま返される。
engine は中身を解釈しない。Ruby 側の `checkpoint!(dir, at:)` が position を、session が
持つ provenance(config digest、dataset、gem / Ruby / engine の版)を合わせて詰める。
session を開かずに読む口が `Torobi::Checkpoint`(manifest / graph_json / position / exist?)。

残る欠けは checkpoint 側ではなく調達側にある: MLX と mlx-c の exact revision が
OminiX のビルド済みバイナリ由来で不明なこと(docs/vendoring.md)。mlx-rs の rev は
`build.mlx_rs` に入っている。

### 11.3 メモリ

Ruby の GC は device memory を把握しない。対策: 中間値を engine の内部に閉じ込める、
Ruby へ返すのは loss・metrics・明示的に copy したテンソルに限る、step 末に state を
まとめて eval する、MLX の active / peak / cache memory を Ruby へ公開する、
1,000 / 10,000 step の plateau テストを持つ。

### 11.4 配布(方針は未確定。ただし feasibility は M1 で確かめる)

選択肢は開いたまま進める:

- arm64-darwin platform gem(拡張ビルド済み。導入は最軽量、リリース工程は重い)
- source gem(要 Rust toolchain + cmake。工程は軽いが導入者に要求が乗る)
- 併用(source を正、platform を利便として)

**ただし決定を遅らせることと、可能性を確かめないことは別である。** 現時点で判明している
制約(`docs/vendoring.md`):

- MLX の exact revision が不明(OminiX のビルド済みバイナリを使っている)
- `mlx.metallib` が約 100 MB あり、**dladdr で拡張バンドルの隣に置く必要がある**
- それが無いとプロセスが死ぬ(§4.1)

よって **M1 の出口条件に「隔離環境へ gem install → require → 1 step」の smoke test を
含める**(§9.1)。開発期は source checkout で進める。Python ランナーは optional な開発依存。

## 12. 検証の層(v1 §18 を本文化)

1. **純 Ruby の構造テスト**: 所有権、位相順序、安定した parameter path、直列化の
   round-trip、決定的 digest、shape / dtype 推論、config 検証。native 不要
2. **engine の Rust 単体テスト**(`rake rust_test`): 境界値型、GraphConfig の受理と拒否、
   batch の束縛、optimizer の算術と slot の追従、freeze 窓、checkpoint の往復と拒否。
   Ruby を通さずエンジン内部の契約を留める層。**`--test-threads=1` で回す**: MLX の
   default stream は単一のコマンドキューで、2 スレッドが同時に投入すると Metal の
   assertion がプロセスごと落とす(拡張が mutex を持つのと同じ理由)
3. **native 境界テスト**(§4.1): 例外変換、panic が C を貫通しないこと、致命の挙動、
   GVL 解放、GC 下での生存、二重 close。**subprocess で回す**
4. **Python MLX との differential**: forward、scalar loss、parameter path ごとの
   gradient、optimizer state、更新後 parameter、checkpoint load 後の最初の step。
   oracle は版付きの成果物として生成し、**生成できない場合は fail closed**(pass 扱いにしない)
5. **数学的テスト**: 小さいテンソルでの有限差分、broadcasting と reduction の property、
   weight sharing 時の勾配加算、softmax / cross entropy の数値安定性、極値・zero-size・
   singleton、dtype ごとの tolerance
6. **収束テスト**: 線形回帰、小さな MLP、tiny dataset の過学習、そして M3b の parity
7. **運用テスト**: memory plateau(1,000 / 10,000 step)、強制 GC、checkpoint の中断と
   resume、スレッドの進行、隔離環境での installed-gem smoke

## 13. リスク台帳(v3.2 で更新)

| Risk | 対処 |
|---|---|
| mlx-rs / MLX の追従 | exact rev pin(Cargo.lock が強制)+ 依存台帳 + 一版ずつの upgrade(全スイート) |
| 狭い腰の劣化(関数が増え広い binding 化する) | セッション API の関数数を計測対象にする。テンソル越境と callback の禁止を review 基準に |
| IR と engine の意味ずれ | 単一 manifest から両面生成、differential test |
| resume / replay 非再現 | TrainState(乱数込み)の明示管理、resume=連続・replay=再演のテスト |
| checkpoint 破損 | manifest + atomic rename + inventory 検証(§11.2) |
| **致命が Ruby に届かない**(初期化失敗でプロセス死) | preflight の拒否リストを増やす、subprocess 異常系テスト、supervisor 前提を文書化(§4.1) |
| **batch 経路の性能** | 計測済み(§5A.2.1): 呼び出しではなく JSON 直列化が支配的。対策は packed encoding であり、投入キューではない |
| **配布(metallib 100 MB と配置制約)** | M1 の installed-gem smoke で早期に確かめる(§11.4) |
| タップの常設によるメモリ / fusion 劣化 | stats 縮約を既定に、full タップは debug 用と明記 |
| 窓能力の際限ない要望 | 能力は列挙制。新規はエンジンの名前付き機能として審査 |
| DSL の自由度肥大 | op registry 制。escape hatch は §10 |

## 14. 関連プロジェクト

- kohagi / kohagi-serve: encoder のサービング、教師採点、パリティ oracle。Torobi の出力
  (HF 配置 fp32 safetensors + 1_Pooling)は無変換で載る。
- burn-rb: 制約付き Graph DSL + native 実行の先行。参考にするが、DSL の書き味の互換は
  目的にしない。教訓は「直接実行の追求ではなく、記述の所有」。
- 本番 LLM サービング: 買う(llama-server / vLLM)。Torobi は merge / GGUF 変換ブリッジまで。

## 15. 次のワークパッケージ(G0: 実行契約)

M0 と M1 の一部(§9.1 の M1 のうち single-step とその境界の初期形)は実装済み。
次は M2 ではなく **G0** から始める。

1. **batch 経路**: `run_step(batch)` を追加し、bindings 固定の現状(セッション開始時の
   入力を全 step で使い回す)を置き換える。可変長 batch のコピー経路と、その境界コストの
   計測(§9.1 M1 の出口条件)
2. **接続の型**: 名前つき model output、`ModelOutputRef`、`BatchRef`、`stop_gradient` op、
   model 名による parameter 名前空間、argnums 規則(§5A.3)
3. **エラー 3 分類の実装**(§5A.4)と、preflight の拒否リストの整備
4. **journal のスキーマ**(§8.5)と replay 2 モードの骨格(§8.6)。実装は M2.5 だが、
   何を記録するかは G0 で決める

### 15.1 G0 の進捗(2026-09-03)

| 項目 | 状態 |
| --- | --- |
| batch 経路 | **済**。`step!(batch)` / `run(batches)`、packed encoding(§5A.2.1) |
| 接続の型 | **済**。名前つき output、`Source`(batch / model output)、`stop_gradient`、model 名の名前空間、`train` 集合と argnums 規則 |
| エラー 3 分類 | **済**。`ConfigError` / `StepError` / `EngineUnavailable`(§5A.4)。4 つ目(abort)は文書とテストで固定 |
| journal スキーマ | **済**。`Torobi::Journal`(JSONL、6 種の entry)と `Provenance`。観測も記録する(§8.5 の訂正どおり) |

### 15.2 M1 の進捗(2026-09-03)

| 出口条件 | 状態 |
| --- | --- |
| 異なる batch での forward / grad / update | **済**(§15.1) |
| FFI 異常系(例外変換、panic、致命、GC、多重セッション) | **済**。subprocess テスト |
| GVL 解放中に別スレッドが進む | **済** |
| 境界コストの実測 | **済**(§5A.2.1)。測って設計判断を覆した |
| installed-gem smoke | **済**。56 KB の source gem がビルドされ、metallib が dladdr の見る場所に入り、checkout の外から 1 step が動く。当初は絶対パス依存のため「このマシンでのみ」だったが、pinned git 依存(§5)に変更し、**ローカル checkout を隠した状態でビルド・インストール・実行が通ることを確認**(`docs/vendoring.md`)|

配布は形も事実も成立した。残るのは方針の決定(§11.4)であり、
それは利用者像が見えてからでよい。

### 15.2.1 外部レビュー(2026-09-03)で見つかった穴と対処

| 指摘 | 対処 |
| --- | --- |
| objective が parameter を持てた / 複数 output / loss 名でない / 非 scalar → engine が辞書順で選び、空 parameter slice を添字して **abort** | GraphConfig が契約を強制(§5A.3)。objective 無しの場合も「単一 model・単一 scalar loss」を要求 |
| optimizer state を欠く checkpoint が restore に成功し、次の step で **panic** | restore は全て読み・検査・eval してから一括 commit。slot の有無は optimizer の要求と照合 |
| step が optimizer → params → RNG を順に書き換えるため、失敗が部分更新を残す | step を transaction 化(§5A.4) |
| checkpoint 置換が「削除 → rename」で、間に停止すると両方消える | 旧を退避 → rename → 旧を削除。書き込み後に全 tensor の読み戻し検証と fsync |
| batch の dtype が境界で失われ、i32(embedding)が表現できない | typed encoding(§5A.2)。`take` op も実装し、embedding の end-to-end を検証 |
| metallib があっても Metal device 不在で abort する環境がある | preflight が **subprocess で probe**(203 ms、プロセスあたり 1 回) |
| journal の digest が `("ab","c") == ("a","bc")` | 長さ framing + canonical JSON。entry ごとに flush、deep freeze |
| `build_info` が実態と違う("path dependency") | build.rs が Cargo.toml の rev を埋め込む |

### 15.2.2 外部レビュー第 2 回(2026-09-03)

「in-process の狭い腰は維持すべきだが、長時間 GPU 学習を同居させる設計としてはまだ穴がある」
という判定。実測で再現し、推奨順に対処した。

| 指摘 | 実測 | 対処 |
| --- | --- | --- |
| `RefCell` を GVL 解放中に保持 → 別スレッドの読みで panic → **Ruby fatal でプロセス終了** | 再現(`RefCell already mutably borrowed`) | `Mutex` + `try_lock` の状態機械。使用中の session は **busy を答えて生き続ける**。blocking lock は GVL 待ちで deadlock するため使わない |
| Rust panic は回復可能な例外ではない(magnus が fatal に変換) | 再現 | GVL ラッパは panic を **返す**(resume しない)。session は **Poisoned** になり以後拒否。プロセスは生存 |
| `run` が全 batch を先に materialize → 無限 enumerable が始まらない、多重保持、Ctrl-C が span 末尾まで効かない | コード上明白 | `run` は **step 単位で Ruby が駆動**(境界コストは実測で誤差)。block を渡せば step ごとに yield。bulk 形は §15.10 で削除した |
| fork 後の安全性を probe では保証できない(`@probe_result` が子に継承される) | — | 拡張ロード時 PID を記録し、不一致なら `EngineUnavailable`。probe の記憶も PID に紐づけ。prefork worker(Puma clustered / Sidekiq / Spring)は非対応と明記 |
| `close` がなく、GPU 資源の解放が GC 任せ | — | 冪等な `close` と `Closed` 状態、block 形式の `ensure`。device memory の観測(`Torobi::Memory`: active / cache / peak / limit、`clear_cache!`、`limit=`)。「開いて閉じた 20 session が MB 単位を残さない」ことをテスト |

**残る推奨**: 長時間学習を `Process.spawn` した専用 runner で走らせることを正式経路にする
(§11.4 の配布方針と同じ判断の裏表)。→ **済**(§15.8、`Torobi::Runner`)。

### 15.3 M2 の進捗(2026-09-03)

| 出口条件 | 状態 |
| --- | --- |
| AdamW が oracle の最初の数 step と一致 | **済**。論文の式から Ruby で書き起こした oracle と 5 step すべて 1e-5 一致。bias correction(最初の step が勾配によらず約 lr 動く)と decoupled decay も検証 |
| RNG state | **済**。session が key を持ち、step ごとに split する。dropout が最初の消費者で、seed の一致・不一致・p=0 の恒等性・同一 step 内の 2 つの dropout が別の mask を引くことを検証 |
| checkpoint / resume(resume = 連続実行) | **済**。manifest + parameters / optimizer / random の safetensors、staging → atomic rename。**dropout を有効にしたまま** step 5 で中断・再開した run が、12 step 連続実行と 1e-5 一致 |

optimizer は engine が所有(§5 の方針どおり mlx-rs のものを使わない)。checkpoint は
digest・path・shape・optimizer 種別を読み戻しで検証し、不一致を拒否する。

### 15.4 M2.5 の進捗(2026-09-03)

| 出口条件 | 状態 |
| --- | --- |
| ノブ | **済**。lr / seed / freeze / unfreeze / put / checkpoint / evaluate。rollback は `restore` がそれであり、専用ノブは足さない。損失重みは M4 まで持ち越す(§15.9) |
| フック | **済**。`s.on(event, every:)` と `s.use(policy)`、発火点は step / span_end / checkpoint_written。柵 4 本(登録順、再入禁止、例外で span 停止、調整は journal 経由)。標準ポリシー 5 つ |
| journal の実運用 | **済**。全ての窓操作が自動記録。header に provenance、entry ごとに flush |
| 観測タップ(B+) | **済**。ノードの安定命名 → `tap(name, stat:)`、norm / mean / extent / full。デバイス側縮約。**タップ有無で学習結果が一致する**ことを検証 |
| 2 種の replay | **済**。`Replay.action`(policy を再実行せず操作を適用、bitwise 一致)と `Replay.rerun`(policy を再実行し、観測値と判断まで照合)。データは journal が digest でしか名指ししないので呼び出し側が供給する |
| freeze を構造変更として | **済**。argnums の変更に optimizer slot が追随(保持 / 破棄 / ゼロ初期化)、step 数は据え置き |

**残る作業**(M3a の前に片付けるか、並行するか): なし。§15.9 を参照。

### 15.5 engine の内部整理(2026-09-03)

916 行の `session.rs` が 7 つの責務を抱えていたのを、境界に沿って割った。

| module | 何を持つか |
| --- | --- |
| `tensor.rs` | 境界を渡る値(`Tensor` / `PackedTensor` / packed の解凍) |
| `plan.rs` | run が何をするか。open 時に確定し、以後動かない(models、paths、candidates、digest、batch の束縛、freeze パターン) |
| `state.rs` | run が何を溜めたか。動くのはここだけで、動くのは `advance` 1 箇所(params、argnums、optimizer、RNG、counters、checkpoint) |
| `executor.rs` | step をどう計算するか。状態を持たない関数だけ(forward / resolve / differentiate) |
| `session.rs` | 上 3 つに至る語彙。261 行 |

Rust 単体テスト 64 件を同時に入れた(§12 の層 2)。書いている最中に、**MLX を 2 スレッドから
同時に叩くと Metal の assertion でプロセスが落ちる**ことが分かった(`Completed handler
provided after commit call`)。拡張側は既に mutex で直列化しているので実害はないが、
cargo のテストハーネスは既定で並列なので `rake rust_test` は `--test-threads=1` を渡す。
あわせて `engine/build.rs` が metallib をテストバイナリの隣(`target/<profile>/deps/`)へ
symlink する。105MB なのでコピーではなく symlink。

### 15.6 checkpoint を run checkpoint にする(2026-09-03)

§11.2 に対して欠けていた 4 点を埋めた。schema version は 1 → 2(未リリースなので
後方互換は取らない)。

| 欠け | 埋めかた |
| --- | --- |
| `graph.json` が無い | GraphConfig のバイト列をそのまま同梱。読み戻しで digest 照合 |
| dtype inventory が無い | manifest の parameter entry に dtype。safetensors 側の実物と突き合わせて拒否 |
| semantics version / 実行機が無い | manifest に `semantics_version` と `platform`(os / arch) |
| epoch・batch position・sampler が無い | **engine には書けない**ので `run` として不透明に運ぶ。`checkpoint!(dir, at:)` が position を、session の provenance が config digest・dataset・版を詰める。`restore` は position を返す |

session を開かずに読む口として `Torobi::Checkpoint` と `Native.checkpoint_manifest` を
足した。どの checkpoint から再開するかを決めるのに、session を 1 つ開く必要はない。

この時点で Ruby 143 件 / Rust 74 件。

### 15.7 session の並行契約(2026-09-03)

`notes/SESSION_CONCURRENCY_SPEC.md` を実装と突き合わせ、Draft を Accepted に直した。
そのとき **既存の欠陥が 1 つ再現した**。

`Timeout` や `Thread#raise` を普通の `run` に掛けると、session が二度と使えなくなる。
CRuby は nogvl 領域を抜けるときに保留中の割り込みを送出し、その longjmp が Rust の
フレームを飛ばす。`MutexGuard` を境界の向こうまで保持していたので drop されず、
ミューテックスが誰にも保持されないまま閉じたままになる。`close` すら通らないので
device memory も GC 待ちになっていた。既存テストが捕まえていなかったのは、span の
中断テストが Ruby の block から例外を上げており、**非同期割り込みと同期例外は別経路**
だからである。

直し方は 3 つ組である。

1. engine を slot から取り出し、使い、戻すまでを **GVL 解放区間の中で完結**させる。
   境界の前には何も保持せず、後には plain value を読む以外に何も残さない。
2. `rb_nogvl(..., RB_NOGVL_INTR_FAIL)` に切り替え、区間の出口で割り込みを見ないように
   する。ただし入口の拒否は `RUBY_VM_INTERRUPTED_ANY` が scheduler の timer でも立つ
   ため大半が空振りなので、`rb_thread_check_ints` を挟んだ上限付き再試行にし、
   最後は GVL 保持のまま実行して必ず終わらせる。
3. Ruby 側で **engine の変更とその journal 記録を `Thread.handle_interrupt` で括る**。
   これが無いと engine は step を取ったのに記録が 1 手遅れ、replay が食い違う。

あわせて `Busy` / `Interrupted` / `SessionPoisoned` / `SessionClosed` を型にし
(`Busy` だけが `StepError` の下: 「engine は拒否したが session は自分のもの」が真な
のはこれだけ)、`step` / `loss` / `lr` / `seed` を snapshot から読むようにした。
進捗表示や metrics のスレッドが `Busy` を受け取る形をやめている。

spec から**落としたもの**: `Arc<SessionCore>` (保持者が 1 つしかない)、`ArcSwap`
(依存を増やす)、cancellation flag と unblock function (`run` は Ruby が step 単位で
駆動しているので Ruby のループ自体が cancellation point)、実行中の非同期 `close`
(single writer 契約の外)。

**レビュー後に足したもの** (同日): Session ごとの排他では足りないことが分かった。
`Mutex<Slot>` は 1 Session に 1 thread しか入れないが、**2 つの Session は同時に MLX へ
入れる**。2 Session × 2 thread × 300 step で 3 回中 3 回プロセスが落ちた
(`commit command buffer with uncommitted encoder`)。MLX を走らせるものは Session を
問わず **process-global なゲート**を取るようにした。GVL を解放した状態でのみ取るので
待っても他の Ruby thread は止まらず、拒否ではなく交代になる。`close` の device memory
解放と `Torobi::Memory` の process-global な呼び出しも同じゲートを通る。

fork guard も native へ移した。Ruby の `Preflight` は新しい `open` しか拒否できず、
**親で作った Session を子で使う**経路と `Torobi::Native` の直接利用を素通りさせていた。
拡張ロード時の PID と Session 作成時の PID を native に持ち、lock を取る前・ゲートに
触る前に確認する (fork 時に他 thread が握っていたゲートは子では永久に閉じているので、
順序が重要である)。子での `close` は engine を drop せず `forget` する。

この時点で Ruby 154 件 / Rust 74 件。

### 15.8 専用 training process(2026-09-03)

§11.4 と review #3 が繰り返し求めていた「長時間学習は `Process.spawn` した専用 runner
で」を `Torobi::Runner` として形にした。spec §9.4 が契約である。

**両者が共有するのはディレクトリだけ**にした。子は journal と checkpoint をそこへ書き、
親はその両方を読む。開いたままのパイプも、版を上げるプロトコルも要らない。進捗を子に
尋ねないのが要点で、子は step の中かもしれず既に居ないかもしれないが、ディスク上の
記録はどちらでも答える(journal が entry ごとに flush しているのが効く)。

| | |
| --- | --- |
| 起動 | `Process.spawn` で exec。fork ではないので Metal device はついて来ない。run ディレクトリは `TOROBI_RUN_DIR` |
| 停止 | 親が TERM。子の handler はフラグを立てるだけで、ループが step 境界で見る。入っている step を終え、最後の checkpoint を書いて 0 で終わる。猶予後に KILL |
| 終了 | 0 = 完走または停止(journal 最後の note が区別)、69 = EngineUnavailable、70 = 例外、**signal = 落下**。signal を番号に混ぜないのは、それがこの仕組みの存在理由だから |
| resume | 既にある checkpoint を消さない。消さないことが resume の経路である |

落下は `Process.kill("ABRT")` をテスト用スクリプトに仕込んで固定した。親が `crashed?`
と見分けられること、**その時点の checkpoint がそのまま読めて resume できる**ことまで
確かめている(checkpoint は rename で置くので半端が残らない)。

作っている途中で `Runner#running?` のバグが 1 つ出た。`waitpid(WNOHANG)` は終了
ステータスを刈り取るので、捨てると `wait` が報告するものを失う。問い合わせが状態を
変える操作だった。

最後に spec を実装と一行ずつ突き合わせ (rev 3)、ずれていた 6 点を直した。所有権モデルに
`origin_pid` とゲートが載っていなかったこと、fork guard を「Ruby の Preflight が持つ」
と書いたままだったこと、§5.1 と §5.2 の順序が逆だったこと、非目標の「fork safety」が
拒否と混ざっていたこと、§10 の番号が飛んでいたこと、そして `Torobi::Interrupted`。

最後のものは実装の方を直した。`RB_NOGVL_INTR_FAIL` の拒否を報告せず再試行に変えた
時点で、この型はどこからも上がらなくなっていた。上げられない型を残すのは呼び出し側に
嘘の分岐を作らせるだけなので、型ごと落とした。

この時点で Ruby 162 件 / Rust 74 件。

### 15.9 残りのノブを検討して、2 つとも入れなかった(2026-09-03)

M2.5 に「rollback ノブ、損失重みノブ」が残っていた。改めて必要性を検討し、**どちらも
足さず、代わりに実際に空いていた穴を 2 つ塞いだ**。

**損失重みノブ → M4 まで持ち越す。** 固定の重みなら `mul_scalar` で既に書ける。ノブが
要るのはスケジュールしたいときだけで、最初の的(310M → 130M reranker)でそれが要るかは
未証明である。正しい形は `{"knob" => "alpha"}` という入力 source で、温度やしきい値付き
op にも効く一般化だが、これは GraphConfig の schema に語彙を足す変更(digest が変わり、
checkpoint に載り、replay が扱う)であり、**使う人がいて初めて形が決まる**。

**rollback ノブ → 足さない。** rollback は `restore` がそれであり、行き先の checkpoint を
指す以上、専用ノブは名前が増えるだけである。ただし調べる過程で `advance` が **loss を見ずに
無条件で commit していた**ことが分かった。非有限な loss の step は勾配も非有限なので、
取ると全 parameter が NaN になり、checkpoint 無しでは戻れない。これが rollback ノブが
plan に載っていた理由そのものである。

そこで **非有限な loss の step は取らない**ことにした。counter と RNG は進める(step は
試みられ、batch は消費され、forward は RNG から引いたので、resume が同じ列を見るには
進める必要がある)。parameter と optimizer slot は動かさない。loss は報告するので hook は
今までどおり発火する。loss は commit の直前に既に eval 済みなので、判定はタダである。

**ただしこれは rollback を不要にはしない。** 実装してテストを書いて分かったこと:

| 症状 | 何が起きるか | 答え |
| --- | --- | --- |
| 悪い batch 1 つ | その step だけ非有限。parameter は健全 | guard だけで足りる。checkpoint 不要 |
| 発散 | parameter 自体が forward を overflow させる位置にある | step が取られないので lr を下げても**何も動かない**。stuck であって corrupt ではない。**戻る先が要る** |

`NaNGuard` は両方を見るようにした: lr を下げ、行き先があれば戻り、非有限が
`patience` 回続いたら諦める。当初 "guard があれば rollback は要らない" と考えたが、
2 つ目の症状で誤りだと分かった。

**もう 1 つの穴: eval。** §8.3 の A のノブ表に `eval` があるのに、forward だけを走らせる
口が無かった。validation を取るには `loss_and_grads` を呼んで gradient を捨てるしかなく、
要らない backward を毎回払っていた。`Session#evaluate(batch)` を足した。gradient を
計算せず、**乱数も引かない**。interpreter は「key が無い = 学習パスではない」と読むように
なり、dropout は恒等になる。key と別に mode フラグを持たせると、その 2 つが食い違いうる。

M3b の「tiny dataset の過学習」も M4 の「実験が完走し記録が残る」も validation を要るので、
これは近いうちに必ず当たっていた。

この時点で Ruby 164 件 / Rust 81 件。

### 15.10 ゲートの穴を 5 つ塞ぐ(2026-09-03)

並行契約のレビューで 5 点。4 つは P1 で、いずれも実在した。

**1. ゲートの迂回。** `Session.open` は parameter・RNG key・optimizer slot を作るのに
ゲートを通っていなかった。`seed=` は setter の顔で RNG key を作り、`fetch` は読み出しの
顔で array を評価して device から copy していた。どちらも「MLX を呼ばない」側に分類
されていた。**呼び出しごとに判断させる形が間違いのもと**なので、既定を「ゲートを通す」に
反転し、通さないものを `on_cpu` として列挙した(plan が持つ名前の読み出し、tap 集合の
変更、host 側 tensor のコピー)。

その副作用で、同一 Session の 2 本目がゲートに並んで**成功する**ようになった。これは
§1 の single writer を黙って崩す(1 つの Session を 2 thread で共有した呼び出し側が、
交互に step を取られる)。ゲートの前に session を claim し、取れなければ `Busy` を返す
ようにした。

**2. GC による解放がゲート外。** `close` を忘れた session を Ruby の sweep が drop すると、
別 thread が MLX の中にいる最中に device memory が解放される。native の `Drop` を足し、
そこでも PID を先に見て、親ならゲート下で解放、fork 子なら `forget` する。

**3. `rb_thread_check_ints` が §0 を再導入していた。** 生で呼ぶと longjmp で送出するので、
そこから Ruby までの Rust フレームの destructor が飛ぶ。engine は slot に残るので session は
壊れないが、retry のために持っている work とそれが借りる batch が漏れる。
`magnus::rb_sys::protect` で包み、longjmp を値に変えた。

**4. journal の完全性を満たさない公開 API。** `run_uninterruptible` は複数 step を 1 回の
native 呼び出しで回すので、**step ごとの entry を書けない**。実測で「a fraction of a
percent」しか効かない最適化のためにその穴を残す理由がないので、engine 側の `run_steps`
ごと削除した。`repeat` も `step!` の繰り返しに直した。engine を進める公開 API は
`run_step` 1 つになった。

**5. JSONL の読み戻し。** entry ごとの flush は 1 行の書き込みが不可分であることまでは
保証しない。`Journal.read` は改行で終わっていない最後の行だけを捨て、途中の壊れた行は
エラーにする。前者は途中で殺されたこと、後者はファイルが壊れたことで、読み進めるのは
記録を捏造することになる。あわせて spec §9.1 の「ゲートは GVL 解放状態でのみ取る」に
例外 2 つ(区間拒否の最後の手段、GC の sweep)を明記した。どちらも待つ相手が GVL を
要らないので deadlock しない。

この時点で Ruby 168 件 / Rust 81 件。

### 15.11 MLX の制約を、MLX を所有する crate へ(2026-09-03)

engine のレビューと `notes/ENGINE_RUNTIME_BOUNDARY_PLAN.md` の提案を合わせて実施した。
提案の要は 1 行である: **MLX を安全に扱う判断を、MLX を実際に所有する crate に 1 回だけ
書く**。

§15.10 でゲートの迂回を 3 つ塞いだが、あれは症状だった。原因は、制約の発生源 (engine が
MLX を持つ) と強制する場所 (extension) が分かれていたことで、その形のままでは API を
増やすたびに同じ漏れが起こる。実際 engine の CLI と engine 自身のテストは、どのゲートも
通っていなかった。

| 移したもの | どこへ |
| --- | --- |
| process-global gate | `engine/src/runtime.rs` の `Runtime`。生成口は非公開で、`OnceLock` 1 つだけ |
| PID guard | 同上。`torobi_engine::initialize()` を拡張ロード時に 1 回。Session ごとの PID は廃止 |
| MLX を呼ぶかの分類 | engine の `Session` facade。`SessionCore` は `pub(crate)` で、Runtime を通らずに MLX へ届く公開経路が無い |
| device memory の解放 | `Runtime::release`。明示的 close と `Drop` が同じ経路を通り、fork 子では `forget` する |
| allocator 操作 | `torobi_engine::memory::Memory`。裸の関数は `pub(crate)` |

extension に残したのは Ruby runtime の制約だけである: 値の変換、GVL の解放、非同期
割り込みを `Error` にすること、同一 Session の 2 本目に `Busy` を返すこと、snapshot。
`on_cpu` と `static MLX` と native の `Drop` は消えた。unsafe な Ruby C API は
`gvl.rs` だけになった。

**一番はっきりした成果は engine のテストである。** 公開 `Session` を経由する 28 件が
既定並列で通るようになった(`rake rust_test:facade`)。移す前は同じコマンドが SIGSEGV
していた。残り 51 件は plan / state / optimizer / tensor の単体テストで、facade を
通らないので今も直列である。**通らないのが正しい**: gate は facade の仕事であって、
テストハーネスのために各層へ降ろすものではない。

実装中に決めたこと 3 つ。

- **panic は Runtime 全体を poison する。** 1 回の panic でそのプロセスの MLX 作業が
  全部止まる。MLX は process-global で、gate 保持中の panic が stream をどう残したか
  分からないためである。`Torobi::RuntimePoisoned` を型として足した。代償は
  「その後 checkpoint も書けない」ことで、答えは §15.8 の専用プロセスである。
- **`catch_unwind` は gate の外に置く。** 中で捕まえると Mutex の poison が発火せず、
  上のルールが空文になる。engine は gate 内で panic を捕まえない。
- **`Session::loss_and_grads` を削除した。** raw な `Array` を crate 外へ渡していた。
  facade を作る作業は、公開 API から MLX handle が漏れていないかの点検も兼ねた。

この時点で Ruby 168 件 / Rust 79 件。

### 15.13 境界の見直しで出た 2 点(2026-09-03)

Ruby と Rust の分担を改めて確認したときに出たもの。分担の原則(Ruby が記述・判断・
記録、engine が実行と MLX の制約、ext は Ruby runtime の制約だけ)は妥当だったので、
穴 2 つだけを塞いだ。

**1. 死ねる経路が 1 つ残っていた。** metallib の不在は Ruby の `Preflight` だけが見て
いたので、`Torobi::Native` を直接使う経路、engine の CLI、engine のテストは素通り
だった。MLX は `dladdr` で自分の位置を引いてその隣を探すので、**同じ問いを同じ方法で
先に engine が聞く**ようにした(`runtime::available`、プロセスに 1 度)。これで
`Runtime::enter` を通る全経路が abort ではなく `EngineUnavailable` になる。
`MLX_METAL_PATH` が設定されていれば、こちらより呼び出し側の方が事情を知っているので
判断を譲る。

境界テストが 1 つ**反転した**。`test_without_the_refusal_a_missing_metallib_ends_the_process`
は「Ruby を経由しなければ死ぬ」ことを固定していたが、今は死なない。
`test_even_the_direct_native_route_is_refused_not_aborted` に置き換えた。

あわせて Ruby 側 `Preflight` から metallib チェックを外し、**probe に専念**させた
(policy を 2 箇所に書かない)。probe が失敗したとき、子プロセスが言ったことを親の
例外メッセージに載せるようにもした。以前は理由を捨てて定型文を出していた。

**2. weights をキーワード引数にした。** `open(config, weights:)`。M3a で model import
(safetensors ファイルからの読み込み)が入ると、パラメータの出どころは複数になる。
positional のままだと String と Hash の型で分岐する形になり、GraphConfig が既に
宣言している initializer から始める第 3 の形も並べにくい。キーワードなら 3 択が
そのまま並ぶ。

```ruby
open(config)                       # 宣言された initializer + seed から(将来)
open(config, weights: hash)        # 値を直接
open(config, weights_file: path)   # ファイルから(M3a)
```

**大きなモデルの weights を JSON で渡す形は M3a では使わない。** batch で「JSON が
1 step の 65%」と実測したのと同じ問題が、130M パラメータでは桁違いの規模で出る。
checkpoint の restore と同じく、パスを渡して engine が safetensors を読む形にする。
ファイルは共有メモリではないので「コピーだけが境界を渡る」は保たれる。

この時点で Ruby 169 件 / Rust 83 件。

### 15.14 M3a の入口: safetensors からパラメータを読む(2026-09-03)

M3a の出口条件は「safetensors ロード」と「1 ブロックの forward / gradient parity」。
その前者を入れた。

**パラメータの出どころを型にした。** `Weights::Inline(json)` と `Weights::File(path)`。
engine が enum で受け、Ruby は `weights:` と `weights_file:` の 2 キーワードで言う。
呼び出し側が型で分岐する形にはしていない(§15.13 の判断のとおり)。

ファイルの tensor 名は **graph が名付けるとおりの qualified path**(`m.l.weight`)
とする。これは checkpoint が書く名前でもあるので、**ある run が到達したパラメータから
別の run を始められる**。蒸留がまさに要る形である: パラメータだけ持ってきて、
optimizer slot も counter も RNG も持ってこない。resume は今までどおり `restore` で、
そちらは全部持ってくる。テストで両者を並べて固定した。

| 検査 | import | restore |
| --- | --- | --- |
| shape | 一致必須。形が違えば別のパラメータである | 一致必須 |
| dtype | **変換する**。bf16 で公開されたモデルは f32 の run を始めるのに何の問題も無い | 一致必須。resume は同じ run でなければならない |
| 欠け | 拒否。どの path が無いか、名前はどう付けるべきかを言う | 拒否 |
| 余り | 無視。大きなモデルの一部を取り込むのは正当 | 該当なし |

HF 配置の名前(`encoder.layer.0.attention.self.query.weight`)からの読み替えは**入れて
いない**。呼び出し側の仕事とし、エラーメッセージにそう書いてある。マッピングを engine
に持たせるのは、実際に読み替えたいレイアウトが 1 つ以上見えてからにする。

残る M3a: 1 ブロックの forward / gradient parity。これは op が足りない
(`config/ops.yml` に 32、`interp.rs` に 20)。layer_norm / softmax / gelu / sdpa が
要る。§15.12 で保留にした「op の事前解決」は、この作業と同じ波でやるのが自然である。

この時点で Ruby 175 件 / Rust 88 件。

### 15.15 M3a: op を open で解決し、1 ブロックを通す(2026-09-03)

**先に構造を直した。** interp は step ごとに op を文字列 match し、属性を
`serde_json::Map` から引いていた。ここに 16 op 足すのは逆方向なので、`engine/src/op.rs`
を作り、`Program::resolve` が open 時に `Op` enum へ落とすようにした。

得たものは速さより**失敗の時期**である。今まで value_and_grad の中から MLX 例外として
step 1 で出ていたものが、open 時の拒否になった。

| 何が | いつ分かるようになったか |
| --- | --- |
| 語彙に無い op | open。ノード番号と op 名つき |
| 属性が無い / 型が違う | open |
| dropout の p が 0..1 の外 | open(乱数を引く前) |
| 入力の数が合わない | open。`config/ops.yml` の arity と照合 |
| 未計算のノードを読む | open。**interp の `values[id]` が範囲内なのはこの検査があるからである** |
| 存在しない input / parameter を読む | open |
| output が無い | open |

**16 op を実装した。** neg / abs / sqrt / exp / log / gelu / relu / sigmoid / tanh /
softmax / slice / sum / layer_norm / rms_norm / rope / sdpa。`config/ops.yml` の 32 と
engine の解決表が**完全に一致**した(差分ゼロを機械的に確認)。

gelu は erf の厳密形にした。tanh 近似ではない。M3b の parity は reference 実装との
一致を見るので、そこで数値がずれる理由を自分で作らない。

**1 ブロックの gradient を有限差分で検証した**(`test/block_test.rb`)。
layer_norm(bias あり) → q/k/v → sdpa → 残差 → layer_norm → GeGLU → 残差 → mse。
oracle は Python MLX ではなく中心差分にした。Python MLX は Torobi を「同じ考えの別の
実装」と比べるが、中心差分は engine の gradient を**engine 自身の forward**と比べる。
autodiff が一致すべき相手はそちらであり、第 2 のツールチェーンも要らず、構成上
fail closed で、この milestone が問うている誤り(forward は正しく backward が違う op)
をちょうど捕まえる。

実測: 勾配の最大が 0.036、解析値と数値差分の最大乖離が **5.35e-06**。閾値は 1e-4 に
した。勾配が全部ゼロでも通ってしまわないよう、勾配の大きさそのものも assert している。

中心差分が捕まえないのは「forward と backward が同じように間違っている」場合で、
それは Python MLX oracle の仕事であり M3b である。

この時点で Ruby 178 件 / Rust 107 件。

### 15.16 M3b の下ごしらえ: reshape と multi-head(2026-09-03)

M3b(ModernBERT)を DSL で書こうとして、**`reshape` が語彙に無い**ことが分かった。
multi-head attention は [batch, seq, heads*head] を [batch, seq, heads, head] にして
から heads を前に出すので、これが無いと 1 head しか書けない。`transpose` も
`handle: true` になっておらず、`linear` の中からしか使えなかった。

`reshape` を 3 面(`config/ops.yml`、Ruby の形推論、engine)に足した。**-1 は 1 つだけ**
許し、そこに残りが入る。symbolic な batch 次元は -1 にしか入れられないので、symbolic な
入力の reshape は -1 を名指すことを要求し、concrete 部分の一致も検査する。形の間違いを
build 時に言うための検査であって、形式ではない。

`transpose` も handle op にした。

**multi-head ブロックが書けて、勾配が中心差分と一致した**。wqkv 1 本 → split →
reshape で heads を出す → transpose で前に出す → rope(q と k のみ) → sdpa →
transpose で戻す → reshape で畳む → wo。39 ノード、6 パラメータ。§15.15 の
single-head と同じ閾値 1e-4 で通っている。

途中で `test_the_manifest_and_the_ruby_side_agree` が落ちた。この検査は shape rule の
一覧を**テストが手書きで持っていた**ので、`Shape::RULES` として `Shape` 自身に宣言させ、
テストは 2 つの実物を突き合わせる形にした。あわせて「どの op も名指さない rule が
Shape にある」ことも見るようにした。誰も検査していない rule は無いのと同じである。

**残る M3b**: ModernBERT 本体(埋め込み + ブロックの積み重ね + local/global の交替)、
tiny dataset の過学習、memory 予算の実測、そして forward / gradient の parity。
最後の 1 つは Python MLX oracle を要り、この機械にはまだ入っていない(§12 の層 4、
§10)。

この時点で Ruby 179 件 / Rust 109 件。

### 15.17 M3b: ModernBERT が公開 checkpoint と一致し、動いた(2026-09-03)

oracle は Python MLX ではなく **kohagi と公開 checkpoint そのもの**にした(§9.2 が
encoder 系の oracle として kohagi を挙げている)。Python MLX はこの機械に入っておらず、
入れずに済むならその方がよい。

**`Torobi::Models::ModernBERT` を書いた。** 数値を埋め込まず、公開の `config.json` を
読んで組み立てる。パラメータの path は **checkpoint 自身の名前**にした。

cl-nagoya/ruri-v3-130m に対して、宣言と実物が**完全に一致**した:

| | |
| --- | --- |
| 宣言したパラメータ | 116 |
| checkpoint が持つテンソル | 116 |
| 片方にしか無いもの | 0 |
| 形の不一致 | 0 |

これは forward を 1 度も走らせずに得られる構造 parity で、アーキテクチャの間違いの
大半はここで落ちる。layer 0 に attn_norm が無いという reference 自身の非対称も含めて
一致している。

**版付き成果物にした。** `test/oracle/ruri-v3-130m.json` が config とテンソル一覧を
記録し、テストはそれと比べる。checkpoint を持たない機械でも回り、再生成は
`rake oracle` で、checkpoint が無ければ**空を書かずに落ちる**(§12)。

**公開 checkpoint の import 経路を直した。** 公開モデルは「このモデルが student と
呼ばれる」ことを知らないので、テンソルにモデル接頭辞が無い。1 ファイルに全モデルを
qualified path で入れる形(run の checkpoint)とは別に、**モデルごとにファイルを指す**
`pretrained: {model => path}` を足した。蒸留はもともとファイルが 2 つある形なので、
これがその入口でもある。

**実物が動いた**(19 層 / 130M パラメータ / 公開の重み、seq 16、batch 1):

| | |
| --- | --- |
| open | 0.65s、active 529 MB(130M × f32 = 520 MB) |
| forward(`evaluate`) | 0.67s |
| step(forward + backward + 更新) | 0.84s |
| peak / cache | 1904 MB / 1412 MB |

M3b の「memory 予算の実測」はこれが最初の数字である。

**残る M3b**: 数値 parity(kohagi の埋め込みと突き合わせる)と、tiny dataset の
過学習。前者はトークン ID を両者で揃える必要があり、kohagi 側に入口が要るかを次に見る。

この時点で Ruby 185 件 / Rust 109 件。

### 15.18 M3b: 数値 parity が出た(2026-09-03)

**oracle は kohagi にした。** candle、CPU、同じ公開の重み、そして人が既に頼っている
出力。Python MLX を入れずに済み、しかも「同じ考えの別の実装」という §9.2 が求める
比較になっている。

**kohagi は変えていない。** `kohagi/tools/reference` を足しただけで、これは kohagi の
`tools/` の既存の規約(「測る・検証するもの。workspace の外、公開クレートに入らない」)
に沿う。tokenizer の 4 行は public でなかったので**再現し**、公開 API を広げていない。
書き出す前に「この ID は kohagi が実際に埋め込んだものか」をトークン数で照合する。

成果物は token ID とベクトルの両方を持つので、Torobi 側に tokenizer は要らない。

**結果**(cl-nagoya/ruri-v3-130m、mean pooling + L2、f32):

| 入力 | トークン | cos | max\|Δ\| |
| --- | --- | --- | --- |
| 瑠璃も玻璃も照らせば光る | 13 | 0.999999970 | 9.3e-08 |
| 犬も歩けば棒に当たる | 7 | 1.000000006 | 1.1e-07 |
| 長文 | **302** | 0.999999966 | 1.1e-06 |

kohagi 自身の `parity_check.py` が「これだけ近い f32 ベクトル 2 本は 1-cos ≈ 1.2e-7 で
f32 演算が飽和する」と書いている。**その飽和点より下**である。

**302 トークンの例が本題である。** 局所窓は 128 なので、そこを超えて初めて local 層
(19 層中 13 層)が global 層と違うものを見る。ここが合うということは、**層種別に
分けたマスクが正しい**ということである。

実は最初のマスク設計が間違っていた。graph が受け取るマスクを 1 つにしていたので、
128 トークン以下では気付けない。kohagi の実装を読んで `local_attention / 2` の
スライディング窓と分かり、`mask` と `local_mask` の 2 入力に分け、両方を組み立てる
`ModernBERT.masks` を足した。302 トークンの例はそれを踏むために選んである。

**M3b の出口条件のうち「全体の forward parity」は満たした。** gradient の parity は
kohagi が backward を持たないので中心差分が担当する(§15.15)。残るのは tiny dataset
の過学習。

この時点で Ruby 187 件 / Rust 109 件。

### 15.19 トークナイザは持たない。線は config が決める(2026-09-03)

「Torobi がトークナイザを持つべきか」を検討し、**持たない**と決めた。代わりに線を
1 段ずらした。

**線: モデルの `config.json` が決めるものは Torobi が持ち、その上流は持たない。**

持たない理由は 4 つある。

1. **別の成果物に属する。** Torobi が読むのは `config.json`(アーキテクチャ)で、
   トークナイザを決めるのは `tokenizer.json`(別のライフサイクル)。同じモデルで
   トークナイザだけ差し替わることも、その逆もある。
2. **native 依存が 2 本になる。** 導入リスクは既に重い(MLX、105MB の metallib、
   dladdr の配置制約、fork guard)。呼び出し側でできることのために失敗経路を倍に
   するのは割に合わない。
3. **バッチの組み立ては throughput の話で、データを持つ側のもの。** 長さでソート
   するか、バケットに分けるか。kohagi はこれに 200 行かけている。§5A.2 の
   「engine は batch を渡されるのであって取りに行かない」が Ruby 側にも効く線。
4. **蒸留でも要らない。** student と teacher が同じトークナイザなら 1 回引いて両方へ
   渡すだけで、違うなら batch フィールドが 2 本になるだけ。

**代わりに足したもの。** `ModernBERT.batch(config, rows_of_ids, seq:)`。pad token も
局所窓も語彙もすべて `config.json` から来るので、呼び出し側に引かせるのは**手元に
あるものを渡していない**だけだった。

パディング方針が複数ある(最長に合わせる / 固定長 / バケット)という懸念は、
**graph が既に `seq` を固定しているので消える**。長すぎる行は切らずに拒否する:
どちらの端を落とすかはデータ側の判断だからである。

**再現性は所有ではなく記録の問題。** `dataset:` は journal の provenance と全
checkpoint の `run` に verbatim で入る。トークナイザの素性はそこに書く、を規範として
明文化した(必須にはしない。spike には書くことが無く、書かせるべきでもない)。
実際にそこへ届くことをテストで固定した。

この時点で Ruby 191 件 / Rust 109 件。

### 15.20 蒸留データは kohagi が作る。教師の構造も一致した(2026-09-03)

「トークナイズは誰がやるか」の答えを実装に落とした。**kohagi**である。理由は
`§15.19` の線ではなく、この案件固有の事情である: 蒸留データは (query, text, 教師の
スコア) で、**kohagi はトークナイザと教師の両方を持つ唯一の道具**だからである。

`kohagi/tools/dataset` を足した(kohagi 本体も公開 API も不変、`tools/reference` と
同じ形)。`{"query":..,"text":..}` の JSONL を読み、`input_ids` と `teacher` を足して
返す。行の他のフィールドは素通しするので、id やラベルが旅の間に消えない。

**一度きりである。** student はハイパラを変えて何度も回すので、310M の forward を
毎 step 毎 run 引き直すのは同じ答えに何度も払うことになる。教師のスコアは float 1 個。
書く前に「この ID は教師が実際に採点したものか」をトークン数で照合する。

実測(ruri-v3-reranker-310m):

| query | text | teacher |
| --- | --- | --- |
| 自転車置き場を増やしてほしい | 駐輪場の増設要望 | **0.7182** |
| 自転車置き場を増やしてほしい | 製品レビュー | 0.0011 |
| 瑠璃も玻璃も…(ことわざ) | その解説 | 0.0007 |

**教師の構造も検証した。** `ModernBERT.classifier` を足し(`ModernBertForSequenceClassification`:
encoder は `model.` の下、その上に pooled head と分類器)、310M の実物と比べた。

宣言 155、ファイル 156、**差は `classifier.bias` 1 つだけ**で、これは正しい。
`config.json` が `classifier_bias: false` と言っており、HF の `from_pretrained` は
フラグから head を組むのでこのテンソルを読まない。kohagi も同じ理由でフラグに従い、
警告を出す(`rerank::bias` にその判断が書いてある)。**checkpoint と config が食い違う
とき config が勝つ**、で三者が一致した。テストはこの 1 つを名指しで許している。

あわせて `slice` を handle op にした(cls プーリングは位置 0 を取るだけで、それが
書けなかった)。

この時点で Ruby 193 件 / Rust 109 件。

### 15.21 混成の重み供給。蒸留が組めた(2026-09-03)

「Ruby だけで学習できるか」を測っていて穴が 2 つ出た。どちらも fine-tuning の本筋に
あるもので、塞いだ。

**1. 公開の配置が 2 種類ある。** bare encoder(`embeddings.*` が root)と分類器
(`model.*` + `head.*` + `classifier.*`)。kohagi も `contains_tensor("model.embeddings...")`
で見分けている。`classifier(config, seq:, encoder_prefix:)` で選べるようにした。

**2. 新しい head はどのファイルにも無い。** ここが本題である。公開 encoder の上に
分類 head を乗せるのが fine-tuning なので、**encoder はファイルから、head は宣言から**
という混成が要る。

「ファイルに無いものは初期化子から」と暗黙にすると、**名前の打ち間違いが静かに
乱数になる**。そこで `fresh:` に「新しいのはこれ」をパターンで名指させ、名指されて
いない欠けは今までどおり拒否する。拒否のメッセージがその区別を説明する。

```ruby
pretrained: { student: "ruri-130m/model.safetensors" },
fresh: ["student.head.*", "student.classifier.*"]
```

engine に `init.rs` を足した(zeros / ones / normal / kaiming_uniform)。IR は前から
初期化子を宣言していたが、engine は読んでいなかった。**seed から引く**ので、同じ seed と
同じグラフは同じ初期値になる(§11.1 が「parameter 初期化 seed」を明示管理の対象に
挙げている、その実装)。パラメータごとに key を split するので、宣言順を変えても
引く値は変わらない。

**実際に蒸留が回った。** kohagi の tool が出した (ids, 教師スコア) を読み、
ruri-v3-130m の encoder + 新しい head、AdamW で 30 step:

    parameters: 119 (116 from the file, 3 new)
    loss 0.397718 -> 0.128939

**「Ruby だけで学習できるか」への答え。** `tokenizers` gem があれば**できる**。教師も
Torobi の中で回せる(§5A.3 の 2 モデル配線がそのためにある)。ただし測った上で、
この案件では勧めない:

| | batch 8 × seq 128 |
| --- | --- |
| 教師 310M の forward | 0.34s / 1261 MB |
| 学生 130M の step | 0.67s / peak 4074 MB |

オンラインは step あたり 1.5 倍だが、決定的なのは倍率ではなく**回数**である。教師の
答えは run をまたいでも変わらないので、10 回回せばオンラインは同じ数字を 10 回計算
する。固定ペアの蒸留はオフラインが素直で、データ拡張や動的サンプリングを入れる
実験ではオンラインしかない。**どちらも今の Torobi で書ける**。

この時点で Ruby 197 件 / Rust 109 件。

### 15.22 OS を落とした。境界に値オブジェクトを入れた(2026-09-03)

計測中に **OS ごと落ちた**。パニックログは `watchdog timeout: no checkins from
watchdogd in 91 seconds`、つまりシステムが 91 秒応答せずカーネルが自分を終わらせた。

原因を追って、前提が違っていたことが分かった。**この機械は MacBook Air M2 / 16 GB**
である(`hw.memsize` = 17179869184)。§15.17 以降の計測は潤沢なメモリを前提に、
**`Torobi::Memory.limit=` も `Torobi::Runner` も使わずに対話プロセスの中で**回して
いた。安全装置を 2 つ作っておいて、作った本人が使っていなかった。

MLX は unified memory なので、教師 1.24 GB + 学生 1.56 GB + 活性 + cache 1.4 GB が
そのままシステム RAM を食う。

**直したもの 1: 境界に値オブジェクトを入れた。**

バッチは `{shape:, data:}` の Hash で書き、`data` は Float の Array だった。マスクは
これで最悪になる: seq 512 / batch 32 の注意マスクは **1680 万個の Float** で、作った
そばから捨てられる。

`Torobi::TensorData`(bytes + shape + dtype)を足した。burn-rb の同名クラスと同じ
規律で、**演算を持たない**。形を変える・結合するはデバイス側の仕事である。違いは
構築子で、`runs` は「同じ値の連なり」を **Array を一度も作らずに** String の掛け算で
書く。マスクとはまさにそれである。

**直したもの 2: マスクの形が間違っていた。**

パディングは「attend される側」の性質なので、クエリ軸は 1 でよい(kohagi の
`padding_mask` がそうしている)。窓は seq と幅だけで決まりデータに依存しないので、
バッチに 1 枚あればよい。

| | 前 | 後 |
| --- | --- | --- |
| padding | `[batch, 1, seq, seq]` | `[batch, 1, 1, seq]` |
| 窓 | 同上(行ごとに複製) | `[1, 1, seq, seq]` を形ごとに 1 枚、記憶する |

足し算は device 上で 1 回。実測(32 × 512):

| | 時間 | 確保オブジェクト |
| --- | --- | --- |
| 前 | 0.850s | 49,451 |
| 後(初回) | 0.001s | 6,598 |
| 後(2 回目以降) | 0.000s | **321** |

kohagi との数値 parity は 3 例とも**まったく同じ**まま(cos 0.999999970 /
1.000000006 / 0.999999966)。

**直したもの 3: `Runner` に上限を渡せるようにした。**

`Runner.new(..., memory_limit: 3_000_000_000)` が `TOROBI_MEMORY_LIMIT` で子へ渡り、
子は session を開く前に `Torobi::Memory.limit=` を設定する。上限を超えると**機械が
黙る代わりに MLX が例外を上げる**。長い run が専用プロセスに住む理由の大きな部分が
これである(§15.8)。

**この件の教訓**は、道具ではなく使い方にあった。§15.8 で「長時間の学習は専用
プロセスで」と書いておきながら、その後の計測を全部対話プロセスでやっていた。
以後の計測は Runner の子で、上限を掛けて行う。

この時点で Ruby 198 件 / Rust 109 件。

### 15.23 burn-rb を読み直して 3 つ取り入れた(2026-09-03)

`lib/burn/` と `ext/burn/src/` を通して読み、Torobi に無いものを探した。

**取り入れたもの**

1. **「拡張のあらゆる公開関数は net を通る」**(`ext/burn/src/error.rs` の
   `boundary`)。Torobi は session の口と process-global な MLX の口は panic を
   捕まえていたが、**`build_info` と `checkpoint_manifest` は素通しだった**。

   正確に言うと、素通しでもプロセスは落ちない。magnus が `method!` の出口で panic を
   捕まえるからである。ただし magnus が作るのは Ruby の **`fatal`** で、これは
   `rescue` で拾えない。**捕まえる価値は abort を防ぐことではなく、`fatal` を
   `Torobi::SessionPoisoned`(拾えて、判断できて、記録できるもの)に変えること**に
   ある。

   `plainly` を足し、入口を 3 つの口に整理した。

   | 口 | 何のため | 何を足すか |
   | --- | --- | --- |
   | `Session::with_engine` | session の engine を使うもの | session の状態機械(busy / closed / poisoned)、GVL、engine の Runtime |
   | `global` | process-global な MLX(`Torobi::Memory`) | GVL、engine の Runtime |
   | `plainly` | MLX に触れないもの(`build_info`、`checkpoint_manifest`) | 捕まえること、それだけ |

   `closed?` と `poisoned?` の 2 つだけは意図的に外にある。この crate 自身のロックを
   取って値を照合するだけで、engine にも MLX にも届かず、panic しうるものが無い。
   口を通しても closure が増えるだけで安全は増えない。

2. **左辺がスカラーの式**(`lib/burn/scalar.rb`)。`1.0 - x` は損失を紙に書くときの
   形なのに、Ruby は `Float#-` が Handle を知らないので書けなかった。burn-rb と同じく
   `coerce` を実装し、返す `Scalar` は「Ruby が次に送る演算子と組でしか意味を持たない」
   ものとして Handle の下に置いた。5 通り(`1.0 - x` / `x - 1.0` / `2.0 * x` /
   `1.0 / x` / `x / 2.0`)を実物の値で検証した。

3. **例外階層を 1 箇所に木で書く**(`lib/burn/errors.rb`)。Torobi の階層は
   `Busy` / `SessionPoisoned` / `RuntimePoisoned` と育っていたのに、全体像がどこにも
   無かった。あわせて burn-rb が明記していた**「Ctrl-C は Ruby の `Interrupt` のまま
   上げる」**を Torobi でも保証として書き、テストで固めた(`rescue => e` が握り潰す
   形にしてはいけない。止められない学習ループは失敗する学習ループより悪い)。

**取り入れなかったもの、とその理由**

| burn-rb | Torobi | なぜ |
| --- | --- | --- |
| unblock function(`without_interruptible`) | 持たない | burn-rb は学習ループ全体を Rust で回すので要る。Torobi は Ruby が step 単位で駆動するので、Ruby のループ自体が中断点になる(`SESSION_CONCURRENCY_SPEC` §2.1) |
| `stash` / `take_pending`(Ruby 例外を Rust フレーム越しに運ぶ) | 持たない | 計算の中に Ruby の callback が入らない設計(§4)なので、そこで Ruby 例外が起きない |
| 例外クラスを Rust 側で `define_error` | Ruby 側で定義し Rust が引く | 純 Ruby の半分(DSL と IR)が拡張なしで同じ階層を持つ必要がある |
| 拡張が生やすメソッドの `@!method` スタブ | 不要 | burn-rb の `Tensor` は拡張クラスそのもの。Torobi は Ruby の `Session` が公開面で、`Native::Session` は内部 |
| in-process の `Device#probe` | subprocess の `Preflight.probe!` | Torobi の失敗様式は例外ではなく abort なので、同じプロセスで聞けない |
| `Train::Config`(宣言してから launch) | `Session#run` | ループの所有権が逆。§15.2 で決めた |

この時点で Ruby 200 件 / Rust 109 件。

### 15.24 型を 1 つにした。`PackedTensor` を消し、出口をバイト列にした(2026-09-03)

境界の型が 3 つあった。engine の `Tensor`(値)、engine の `PackedTensor`
(バイト列)、Ruby の `TensorData`(バイト列)である。うち `PackedTensor` は
「入る方向に 1 回だけ使われる変換関数が、型の帽子をかぶっているもの」だった。
`Tensor::from_bytes` / `Tensor::as_bytes` に変えて消した。話はこうなる。

> **`Tensor` が境界を渡るもの(バイト列として)であり、engine が扱うもの
> (値として)でもある。`TensorData` は同じ 3 つ組の Ruby 側の名前。**

出口も揃えた。`fetch` / `gradients` / `tapped` は `{shape:, data:}` の Hash では
なく `TensorData` を返す。数値になるのは `to_a` を呼んだときだけで、そこに費用が
見える。

**測ったもの**(ruri-v3-130m、`m.embeddings.tok_embeddings.weight`、
52,428,800 値 = 200MB)

| | RSS | 時間 |
| --- | --- | --- |
| 変更前(Ruby Array を返す) | +600MB | 0.142s |
| 変更後(`TensorData` を返す) | +400MB | 0.101s |
| その `to_a` まで呼んだとき | +490MB | +0.278s |

+400MB は 200MB のコピーが 2 回である。MLX の array から `Vec<f32>` へ、そこから
Ruby の String へ。前者はすぐ解放されるが RSS には残る。3 回目(`Vec<f32>` から
`Vec<u8>`)は `as_bytes` が借用を返すことで消した。`&[f32]` を `&[u8]` として読む
のは bytemuck の `cast_slice` と同じで、u8 に整列の要求が無く f32 に不正なビット列が
無いから成り立つ(依存は足していない)。

**ついでに直したもの**: `weights:`(inline)は JSON なので `TensorData` を渡されたら
そこで数値に展開する。これで `s.fetch(p)` の結果をそのまま次の run の初期値にできる。
inline が小さいもの専用の道であることは変わらない。

`plan.rs` のテストが 1 件、赤いまま残っていた。エラー文言を書き直したときに
assertion を追随させ忘れたもので、この変更とは無関係。文言を直した。

この時点で Ruby 200 件 / Rust 109 + 28 件、kohagi との forward 一致は変わらず。

### 15.25 M3b の出口条件を全部満たした(2026-09-03)

残っていたのは 2 つ、gradient parity(全体)と tiny dataset の過学習。

**1. 全体の gradient parity**(`test/modern_bert_gradient_test.rb`)。block 1 つの
中心差分は §15.15 で済んでいたが、**積み重ねが増やすもの**は別にある。埋め込み表、
global と local の交替、2 つの mask、pooled head。ここで間違っていれば他のどこでも
間違えない。

小さな ModernBERT(hidden 8、3 層、heads 2、`local_attention: 4`、seq 6)を作り、
**全パラメータ**の抜き取り位置で解析勾配と中心差分を比べた。層 0 は global、1 と 2 は
local で、窓が実際に何かを切り落とす大きさにしてある。

**刻み幅は掃いて決めた**。差は h² で落ちる(3e-2 で 4.9e-4、1e-2 で 5.5e-5、5e-3 で
1.4e-5)が、そこから先は 2 回の forward の f32 丸めが効いて再び増える(1e-3 で
1.8e-5)。谷は 3e-3 で、そこでの最大差は **4.7e-6**(最大勾配 0.20)。許容は 1e-4 に
した。

**2. tiny dataset の過学習**を 2 つの大きさで。

小さい方はテストの中(4 行、AdamW 200 step で 0.413280 → 0)。これは「配線が
通っている」ことの証明で、checkpoint が無い環境でも走る。

大きい方が本番である(`bench/overfit.rb`)。**公開 ruri-v3-130m の encoder + 新しい
head**、kohagi が採点した 8 行、seq 24:

    119 parameters, 8 rows, seq 24
    step   0: loss 0.27105618
    step  10: loss 0.00006129
    step  30: loss 0.00006799
    9.01s for 30 steps (0.300s each), peak 3892 MB

損失より意味があるのは行ごとの答えの方である。tap で head の出力を読み、その
sigmoid を教師の点と並べた。

| row | student | teacher | | row | student | teacher |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.9982 | 0.9997 | | 4 | 0.9986 | 0.9802 |
| 1 | 0.0005 | 0.0025 | | 5 | 0.0026 | 0.0007 |
| 2 | 0.9983 | 1.0000 | | 6 | 0.9985 | 0.9968 |
| 3 | 0.0023 | 0.0094 | | 7 | 0.0016 | 0.0133 |

**全行 0.012 以内**。損失が 6e-5 で止まるのは sigmoid の裾であって当てはめの限界では
ない(0.9997 は logit 8 を要求する)。

**書き方も含めて出口条件のうち**。`bench/overfit.rb` は自分を子プロセスとして
Runner で起動し、8GB の cap をかける。§15.22 で OS を落としたのは道具ではなく使い方
だったので、**測定スクリプト自体が正しい使い方の見本**であるべきだと考えた。親は 8 行
である。

途中で tap の名前を間違えた(`student.classifier`)。engine は node 名を model で
名前空間化しないので、正しくは `classifier`。**Runner がそれを exit 70 と journal の
note として報告した**のは、この仕組みが意図どおり動いた例でもある。

| M3b の出口条件 | どこで |
| --- | --- |
| 全体の forward parity | §15.18(kohagi と cos 0.99999997) |
| 全体の gradient parity | ここ(中心差分と 4.7e-6) |
| tiny dataset の過学習 | ここ(小: 0.413 → 0、大: 全行 0.012 以内) |
| memory 予算の実測 | §15.17 / §15.21 / ここ(peak 3892 MB) |

この時点で Ruby 203 件 / Rust 109 + 28 件。

### 15.26 配布 smoke を手作業から `rake smoke` にした(2026-09-03)

§15.2 が「済」と書いた installed-gem smoke は、その時に手で 1 回通したものだった。
M3b までに gemspec も extconf も metallib の置き場所も動いているので、**繰り返せる形**
に直した。

    rake smoke   # gem build -> 専用ディレクトリへ install -> require -> 1 step

`tools/smoke.rb` が中身で、テストしているのは packaging であってライブラリではない。
`spec.files` が `require "torobi"` の行き先を全部並べているか、拡張が `target/` に
残っているものではなく**同梱ソースから**建つか、105MB の metallib が bundle の隣に
入るか(MLX は `dladdr` で探すので隣以外は無い。外すとプロセスが落ちる)。

**bundler は「別を指す」ではなく「環境から外す」必要がある**。RUBYOPT・RUBYLIB・
BUNDLE_* の複数から子に届くので、1 つ残ると checkout が load path に戻る。
`Bundler.with_unbundled_env` で外し、`tools/smoke.rb` 自身も**読み込み元が checkout
なら中止する**。Rake の `sh` に env の Hash を先頭で渡す形は効かない(`sh` は末尾の
Hash を自分のオプションとして取る)ことも、途中で分かった。

通った(`torobi 0.0.1`、metallib 105,160,566 バイトが bundle の隣、1 step で
10.625000 → 1.449985)。ただし**このマシンの cargo キャッシュは温かい**ので、
これが言えるのは packaging についてであって、白紙の環境での MLX ビルド時間ではない。

あわせて §9.1 の表の印を現状に合わせた(G0 から M3b まで ✅、M4 が現在地)。

### 15.27 checkpoint の読み戻しは残す。0.4% だった(2026-09-03)

§15.12 で「実測してから決める」と保留していたもの。`checkpoint::write` は書いた
ものを読み戻してから名前を渡すので、130M ではパラメータ 520MB と AdamW の
モーメント 2 本 (合わせて 1GB) をもう一度読むことになる。

測った (`bench/checkpoint.rb`、ruri-v3-130m、AdamW、seq 32):

| | |
| --- | --- |
| `checkpoint!` | 1.921s (最初)、1.477s (前のものを置き換える場合) |
| `restore` | 0.669s |
| ディスク上 | 1512 MB |

読み戻しの実測値そのものは取れないので `restore` を上界として使った (同じファイルを
読み、同じ検査をし、その上で反映まで行う)。**checkpoint 1 回の 45% 以下**である。

**判断: 残す。** 200 step ごとに checkpoint を取る run で、checkpoint 全体が 0.95%、
読み戻しはそのうち 0.43% 以下でしかない。しかもこの step は batch 1 の 0.780s で
測っており、現実の batch では step の方が重くなるので**割合はさらに下がる**。
0.4% で買えるのは「読み戻せない checkpoint が名前を取ることは無い」という保証で、
これは resume がすべて依存している性質である。

**ノブにもしない。** 0.4% のために付けた off スイッチは、いつか誰かが off にする。
§15.9 で 2 つのノブを入れなかったのと同じ理由による。

### 15.12 レビューの残りを片付ける(2026-09-03)

engine のレビューで 🟡 に残していたものを、Runtime の移動と同じ波で処理した。

| | 直し方 |
| --- | --- |
| dtype 語彙が `tensor` と `checkpoint` に分かれ、**内容もずれていた** (checkpoint は `f16` を書けるが graph はその名前を解決できない) | `tensor::VOCABULARY` を単一の表にし、`dtype_named` / `dtype_spelling` を両向きの読み出しにした。語彙外の dtype は綴りを発明せず、checkpoint 書き込みを**拒否する** |
| `checkpoint::write` が 100 行の手続き | `Manifest::of` / `lay_out` / `publish` に分けた。`write` は「並べる、読み戻す、名前を渡す」の 3 行になった |
| `state::restore` が検証と commit を同居させ、不変条件がコメントでしか表現されていない | `accept` が `Restored` を返し、`restore` はそれを commit するだけになった。**全部の検査を通らないと `Restored` は存在しない**ので、不変条件が型の事実になった |
| `Plan` が「open で確定」と言いながら `input_names` を step ごと、`node_names` を 1 関数内で 2 回計算していた | 両方を `open` で確定してフィールドに持つ |
| `TrainState` の `params` / `argnums` / `rng` が pub で、`Session` が 3 つ掴んで executor を組み立てていた | `Pass<'a>` ビュー 1 つに畳み、3 フィールドを private に戻した。`differentiate` の引数も 6 個から 4 個へ |
| `Session::read_manifest` が session の関連関数 | `checkpoint::read_manifest_json` へ移した |

`interp` の op 事前解決 (文字列 match と JSON 属性参照が step ごと) と、checkpoint
読み戻しのコストは M3a / M3b の判断として残している。前者は model import で
ノード数が 3 桁になってから、後者は実測してから決める。**両方とも片付いた**
(§15.15 と §15.27)。
