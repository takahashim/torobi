# Torobi(とろ火)実装計画 v3.1

- Status: Proposal(v3 を外部レビューと介入モデルの議論を受けて改訂)
- Date: 2026-09-03
- Primary target: Apple Silicon macOS
- 名称: gem / リポジトリ = `torobi`、エンジン crate = `torobi-engine`、namespace = `Torobi::`
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

v1 の GraphConfig 設計(§6)、op registry manifest(§7)、乱数と checkpoint(§13〜14)、
correctness 戦略の骨格(§18)は引き続き継承する。

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
- スループット競争、CUDA / Linux(MLX の CUDA backend 成熟後に再訪)
- v1 の non-goals(Python API 互換、Marshal 等)は維持

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

この構成により、v1 で計画の過半を占めた native 境界の設計・試験は
「一度書いて閉じる有界な問題」になる。ASan 等の検査は通常の Rust / 拡張テストとして行う。

なお、プロセス分離は要件ではない(v3 の「同居させない」原則は撤回済み)。腰の狭さも
教義ではなく費用対効果の姿勢であり、価値が実証されれば面を広げる選択(例: 読み取り専用の
限定的な eager API)を妨げない。ただし広げる際は §12「狭い腰の劣化」の審査
(テンソル越境と callback の禁止を破らないか、関数数の増分に見合う価値か)を通す。

## 5. 依存と vendoring(v3 から変更なし)

- torobi-engine は mlx-rs を選択的に vendoring(array / ops / fast / transforms / io。不要部は刈る)。
- nn / optimizer はエンジン内で所有(mlx-rs は参考)。AdamW は oracle の数 step と照合する。
- vendoring 台帳(元コミット、刈った物、MLX / mlx-c / mlx-rs の exact revision)を維持し、
  `Torobi.build_info` が全 revision を報告する。更新は一版ずつの明示作業。

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

### 8.5 journal と replay

窓での操作(ノブ、put、データ切替)は名前付きで journal に自動記録され、
`torobi replay` が同一 seed から決定的に再演する。読み取り(A の読み、B+ タップ)は
記録不要(再演で再現される)。安定したセッションプログラムは recipe(宣言)へ昇格できる。

## 9. 北極星と検証

### 9.1 マイルストーン(v3 から微修正)

| | 内容 | 出口条件 |
|---|---|---|
| M0 | 純 Ruby DSL(2 部 GraphConfig、自動命名、objective の語彙) | native 無しで全テスト、同一定義 → 同一 digest |
| M1 | エンジン最小(線形回帰)+ 狭い腰の拡張 | oracle と forward / gradient 一致、GVL ラッパ経由で 100 step |
| M2 | 学習核 + 窓の契約(スパン、ノブ、フック、journal / replay、checkpoint / resume) | AdamW が oracle と一致、resume = 連続実行、replay = 再演一致 |
| M3 | semantic ops + pretrained import + 最初のアーキテクチャ(ModernBERT) | base-v2 の logit が kohagi-rerank と一致 |
| M4 | 蒸留初号機(310m 教師 → 130m 級) | 自前評価で素の base-v2 と同等以上(対照実験込み) |
| M5 | embedder FT(pooling / InfoNCE / Matryoshka) | kohagi --text と cosine ≈ 1.0、--dims 256 評価 |
| M6(将来) | decoder 系アーキテクチャの追加 + LLM LoRA(1〜3B)、量子化 op、varlen | M5 Pro クラスで一晩以内 |

### 9.2 数値評価の 3 軸

fused kernel と decomposition、実装間の比較は次を**別軸で**評価する(「速くて正しい」を
一括りにしない):

1. **semantic equivalence**: 数学的に同じ関数か(FlashAttention 系は exact)
2. **numerical tolerance**: 演算順序由来の差を dtype ごとの許容誤差で判定
3. **performance**: モデル・系列長・ハードごとに実測(一般化した倍率を主張しない)

oracle は Python MLX(fail closed、版付き成果物)と kohagi(encoder 系)。
有限差分・収束テスト・memory plateau・resume 一致は v1 §18 の構成を継承する。

## 10. Python MLX の二役(oracle かつ遊び場)

GraphConfig を Python MLX で実行するランナーを、検証専用でなく**公式の脱出口**とする。
define-by-run の自由(任意 forward、custom VJP、カーネル実験)が必要な瞬間はここで実験し、
収束したらエンジンの機能・DSL の語彙へ戻す。GraphConfig が両世界の共通言語。

## 11. 配布(方針は未確定。M2 以降、利用者像が見えてから決める)

選択肢を開いたまま進める:

- arm64-darwin platform gem(native 拡張ビルド済み。導入は最軽量、リリース工程は重い)
- source gem(要 Rust toolchain + cmake。工程は軽いが導入者に要求が乗る)
- 併用(source を正、platform を利便として)

開発期は source checkout で進める。Python ランナーは optional な開発依存(不変)。

## 12. リスク台帳(v3 から更新)

| Risk | 対処 |
|---|---|
| mlx-rs / MLX の追従 | exact pin + vendoring 台帳 + 一版ずつの upgrade(全スイート) |
| 狭い腰の劣化(関数が増え広い binding 化する) | セッション API の関数数を計測対象にする。テンソル越境と callback の禁止を review 基準に |
| IR と engine の意味ずれ | 単一 manifest から両面生成、differential test |
| resume / replay 非再現 | TrainState(乱数込み)の明示管理、resume=連続・replay=再演のテスト |
| checkpoint 破損 | manifest + atomic rename + inventory 検証(v1 §14) |
| タップの常設によるメモリ / fusion 劣化 | stats 縮約を既定に、full タップは debug 用と明記 |
| 窓能力の際限ない要望 | 能力は列挙制。新規はエンジンの名前付き機能として審査 |
| DSL の自由度肥大 | op registry 制。escape hatch は §10 |

## 13. 関連プロジェクト

- kohagi / kohagi-serve: encoder のサービング、教師採点、パリティ oracle。Torobi の出力
  (HF 配置 fp32 safetensors + 1_Pooling)は無変換で載る。
- burn-rb: 制約付き Graph DSL + native 実行の先行。参考にするが、DSL の書き味の互換は
  目的にしない。教訓は「直接実行の追求ではなく、記述の所有」。
- 本番 LLM サービング: 買う(llama-server / vLLM)。Torobi は merge / GGUF 変換ブリッジまで。

## 14. 最初のワークパッケージ(M0)

1. `torobi` リポジトリと gem skeleton(純 Ruby から開始)。
2. GraphConfig(model + objective)の Data 定義、deep-freeze、決定的 JSON / digest。
3. op manifest(初期セット + semantic ops)と Ruby 側 registry / 自動命名。
4. Linear lowering、shape 推論、所有権検証。objective graph の最小語彙(mse、重み付き和)。
5. ここまで native もエンジンも無しでテストが通ること。
6. 並行して torobi-engine crate の M1 スパイク(線形回帰、oracle 照合)。
