# Torobi QLoRA 対応 仕様と計画

- Status: Proposal(着手前。§15.50 LoRA、§15.51 bf16 の続き)
- Date: 2026-11-08
- 対象: `docs/plan.md` の M6 に「量子化 op」として据え置かれていたもの
- 前提: Ruby が言語を所有し、Rust は実行だけを所有する(plan.md の原則)

## 0. 要約

LoRA(§15.50)は engine に 1 行も足さずに済んだ。凍結した base、小さな行列 2 つ、
matmul、どれも既にあったからである。QLoRA はそうはいかない。**base を 4 bit で持つ**
という表現そのものが新しく、engine に演算が 1 つ(`quantized_matmul`)、dtype が
1 つ(`u32`)、読み込み時の変換が 1 つ(bf16 → 4 bit)要る。それでも足すものは小さく、
MLX 側の部品は全部揃っていることを確かめた(§2.3)。

目的は「32B 級の dense モデルに LoRA を当てる学習を 48 GB の Apple Silicon で、
7B 級を 16 GB の NVIDIA で」である(§1.2 の表)。bf16 で持つ限り 32B は 64 GB で
どちらにも載らない。

計画は 4 段で、各段の終わりに「何が言えるようになったか」をテストで固定する(§5)。
Phase 0 は MLX の量子化演算を Rust から呼べることと数値が合うことの確認だけで、
IR には触れない。Phase 1 で公開済みの 4 bit checkpoint(mlx-community の `*-4bit`)を
読んで LoRA を当てられるようにし、Phase 2 で bf16 checkpoint を読み込み時に量子化する。
export は Phase 3。MoE(`gather_qmm`)と 4 bit 以外の mode は据え置く(§7)。

## 1. 目的

### 1.1 何を可能にするか

`Torobi::LoRA` は base を**宣言した dtype のまま**メモリに置く。§15.51 で bf16 に
できたので 1 パラメータ 2 バイト、それが下限である。QLoRA(Dettmers et al. 2023)は
base を 4 bit に量子化して凍結し、adapter だけを bf16 で持って学習する。forward では
量子化された重みをそのまま使って行列積を取り(`quantized_matmul`)、backward は
adapter と活性だけに流れる。base の勾配は要らないので、量子化重みの微分は不要である。

Torobi にとっては「`linear` の中で `matmul(x, Wᵀ)` の代わりに
`quantized_matmul(x, W_q, scales, biases)` を発行し、3 つのパラメータを `trainable: false`
で宣言する」という話に収まる。adapter 側(`LoRA#contribution`)は `x.dtype` で A, B を
作るので、そのまま bf16 で動く。

### 1.2 どのモデルがどこに載るか

affine 4 bit、group_size 64 のとき、1 重みあたり 4 bit + (scale 16 bit + bias 16 bit) / 64
= **4.5 bit**。bf16 の 16 bit に対して 0.28 倍である。

| モデル | パラメータ | bf16 | 4 bit g64 | 備考 |
| --- | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B | 0.49 B | 0.92 GiB | 0.26 GiB | テスト用。既存の LoRA テストと同じ実物 |
| Qwen2.5-7B / Llama-3.1-8B | 7.6 / 8.0 B | 14.2 / 15.0 GiB | 4.0 / 4.2 GiB | RTX 3080 16 GB で活性と adapter を含めて載る見込み |
| Qwen2.5-32B / llm-jp-4.1-33b | 32.8 / 33 B | 61 GiB | **17.2 GiB** | Mac 48 GB で載る。16 GB NVIDIA には載らない |
| llm-jp-4.1-32b-a3b(MoE) | 32 B | 60 GiB | 17 GiB | **対象外**。expert の `gather_qmm` と MoE の記述が別に要る(§7) |

活性と optimizer state は別勘定である。adapter の optimizer state は小さい(0.5B rank 8
で 54 万個、§15.50)。活性は seq × batch × 層数で決まり、Torobi には gradient
checkpointing が無いので、長い系列は batch 1 で測ってから言う。

### 1.3 なぜ今か

- 書き手モデルの計画(別リポジトリ llm-bench、SPEC.md Phase 1)で 32B 級への LoRA が
  選択肢に入っており、bf16 では手元の機材のどれにも載らない。
- MLX 0.32.2(現在の pin)に `quantize` / `dequantize` / `quantized_matmul` /
  `gather_qmm` が揃っていて、CUDA backend にも GPU カーネルが入っている(§2.3)。
  上流が追いつくのを待つ段階は過ぎている。

## 2. 現状

### 2.1 ある

| 部品 | どこに | 何をするか |
| --- | --- | --- |
| `Torobi::LoRA` | `lib/torobi/lora.rb` | `rank` / `alpha` / `on:`。`wraps?(path)` は最後のセグメントで照合、`contribution` が A, B を `x.dtype` で作る、`fresh(config)` が実パスの一覧を返す |
| `g.adapting(adapter) { }` | `lib/torobi/dsl/builder.rb` | adapter をスコープに置く。`param` が `@adapter.trains?(path)` で trainable を落とす |
| `g.linear` | `lib/torobi/dsl/layers.rb` | `param("#{name}.weight", [d_out, d_in], dtype: x.dtype)` → `matmul(x, wᵀ)` → bias → adapter の寄与 |
| `causal_lm(config, adapter:, dtype:)` | `lib/torobi/models/llama.rb` ほか | モデル記述は `adapting adapter do ... end` を 1 回書くだけ |
| dtype 語彙 | `lib/torobi/ir/dtype.rb` `ALL = %i[f32 bf16 i32 bool]`、`engine/src/tensor.rs` | engine の `Dtype` enum(`engine/src/mlxc/array.rs:99`)には `Uint32` が既にある。IR 語彙に無いだけ |
| 読み込みと dtype 変換 | `engine/src/plan.rs:469-503` | `Source::Pretrained` はファイルの配列を `shaped()`(形が一致することを要求)で受け、宣言 dtype と違えば `as_dtype` |
| 学習状態 | `engine/src/state.rs` | `params: Vec<Array>` に全パラメータ、`argnums` に微分する位置。凍結は argnums の出し入れ |
| checkpoint | `engine/src/checkpoint.rs` | `parameters.safetensors` に全パラメータを書く。manifest に path / shape / dtype / trained |
| export | `lib/torobi/export.rb` `refuse_adapted!` | adapter つきの run は書かずに断る。merge は未実装(「checkpoint が両方を持っているので失われない」) |
| op 語彙 | `config/ops.yml`(単一の真実)→ `lib/torobi/shape.rb`(形規則)、`engine/src/op.rs`(enum と parse)、`engine/src/interp.rs`(実行) | 属性つき op の前例: `cast` の `dtype`、`sdpa` の `scale` / `causal` |
| 勾配 | `engine/src/mlxc/transforms.rs` | `mlx_value_and_grad`。backward は MLX の VJP。手書きの backward は無い |

### 2.2 ない

- 量子化演算の束縛。`engine/src/mlxc/ops.rs` の `unary!` / `binary!` マクロ群に
  `mlx_quantize` 系は無い。`mlx_optional_int`(`{int value; bool has_value}`)と
  `const char* mode` を渡す束縛もまだ無い(`CString` の前例は `io.rs` にある)。
- IR 語彙に `u32`。packed weight は `uint32` の `[d_out, d_in × bits / 32]`。
- 1 つの重みが 3 つの配列(weight / scales / biases)になる、という形のパラメータ。
- `fetch` / `put` の境界は i32 と f32 の 2 つしか運ばない(§15.51)。u32 は運べない。

### 2.3 MLX 側は揃っている(確認済み)

pin されている prebuilt(`~/.cache/torobi/mlx-prebuilt/v0.6.0.1/mlx-v0.32.2-mlxc-c74db5307cc8…-linux-x86_64-cuda12`)の
ヘッダと静的ライブラリで確かめた。

mlx-c `ops.h`:

```c
int mlx_quantize(mlx_vector_array* res, const mlx_array w,
                 mlx_optional_int group_size, mlx_optional_int bits, const char* mode,
                 const mlx_array global_scale /* may be null */, const mlx_stream s);
int mlx_dequantize(mlx_array* res, const mlx_array w, const mlx_array scales,
                   const mlx_array biases /* may be null */,
                   mlx_optional_int group_size, mlx_optional_int bits, const char* mode,
                   const mlx_array global_scale /* may be null */, mlx_optional_dtype dtype,
                   const mlx_stream s);
int mlx_quantized_matmul(mlx_array* res, const mlx_array x, const mlx_array w,
                         const mlx_array scales, const mlx_array biases /* may be null */,
                         bool transpose, mlx_optional_int group_size, mlx_optional_int bits,
                         const char* mode, const mlx_stream s);
int mlx_gather_qmm(...);   /* MoE 用。本計画では使わない */
```

`libmlx.a`(CUDA 12 版)のシンボル:

- `mlx::core::cu::affine_quantize<__nv_bfloat16, 32|128, 2..8>` — 量子化の GPU カーネル
- `qmm_sm80_mainloop` / `make_qmm_sm90_kernel` — `quantized_matmul` の GPU カーネル。
  RTX 3080 は sm86 なので sm80 経路
- `fp_quantize_dequantize<float|f16|bf16>`、`CublasQQMM` — `mxfp4` / `mxfp8` / `nvfp4` 用
- mode 文字列: `affine`, `mxfp4`, `mxfp8`, `nvfp4`

つまり**ヘッダとカーネルはある**。Linux で実際に走ることと数値が Metal と合うことは
Phase 0 で確かめる(§5)。

### 2.4 MLX の量子化の形(affine)

mlx-lm の `QuantizedLinear` と同じ。`W: [d_out, d_in]` を `group_size` 個ずつまとめて

- `weight`: `uint32 [d_out, d_in × bits / 32]`(packed)
- `scales`: `W.dtype [d_out, d_in / group_size]`
- `biases`: `W.dtype [d_out, d_in / group_size]`

`d_in` は `group_size` で割り切れなければならない。`quantized_matmul(x, weight, scales,
biases, transpose: true)` は `x @ dequantize(W)ᵀ` を packed のまま計算する。
mlx-community の `*-4bit` checkpoint はこの 3 つを `{module}.weight` / `.scales` / `.biases`
の名で、`config.json` の `quantization: {group_size, bits}` と一緒に配っている。
Torobi の `linear` が PyTorch 配置 `[d_out, d_in]` を選んだのと同じ理由で、
この綴りをそのまま使う: **ファイルの名前と Torobi の名前が 1:1 で、renaming が要らない**。

## 3. 設計

### 3.1 量子化重みは 3 つのパラメータである(1 つの属性ではない)

選択肢は 2 つあった。

- (a) `ParameterSpec` に「量子化済み」の属性を持たせ、engine が裏で 3 配列を持つ
- (b) `{name}.weight` / `.scales` / `.biases` を **3 つの `ParameterSpec`** として宣言する

(b) にする。Torobi は「パラメータ = 1 配列」を至る所で前提にしている: `state.params`
の 1 要素、`fetch` / `put` の単位、checkpoint の manifest の 1 行、freeze の pattern の
照合単位。(a) はこれら全部に分岐を足す。(b) は何も変えず、mlx-community の checkpoint
とも名前が 1:1 で、`shaped()` の形の照合がそのまま通る。

代償は、`fetch("model.layers.0.q_proj.weight")` が u32 の packed を指すようになること。
これは §3.6 で扱う。

### 3.2 engine に足すのは op 1 つ、dtype 1 つ

| 追加 | どこ | 内容 |
| --- | --- | --- |
| dtype `u32` | `ir/dtype.rb` の `ALL`、`tensor.rs` の綴り表、`config/ops.yml` の型 | 学習には現れない。量子化重みの packed にだけ使う |
| op `quantized_matmul` | `config/ops.yml`、`shape.rb`、`op.rs`、`interp.rs` | inputs 3 つ(x, weight, scales)+ 任意で biases、attrs `group_size` / `bits` / `mode`、`transpose` は常に true |

`quantize` と `dequantize` は **IR の op にしない**。学習グラフに現れないからである。
`quantize` は読み込み時の変換(§3.4)で、`dequantize` は export と `fetch`(§3.6)で、
どちらも engine の内部呼び出しで済む。IR に無いものは digest にも現れない。

勾配について。MLX の `QuantizedMatmul::vjp` は **x についてだけ**定義されていて、
weight / scales / biases について微分しようとすると例外になる。Torobi では 3 つとも
`trainable: false` で宣言されるので argnums に入らず、autodiff がそこに辿り着くことは
無い。ただし「宣言すれば何でも trainable にできる」を残すと、step で MLX の例外として
出てくる。**`GraphConfig` が構築時に断る**(loss が f32 でないときと同じ扱い、§15.51)。
量子化パラメータを trainable にした宣言は、直し方を名指しした `ConfigError` になる。

### 3.3 shape 規則

```
quantized_matmul(x: [..., d_in], weight: u32 [d_out, d_in × bits / 32],
                 scales: [d_out, d_in / group_size], biases: 同)
  → [..., d_out]
```

`shape.rb` で確かめること: `weight` が u32、`x.dtype == scales.dtype`、
`d_in == weight.shape[1] × 32 / bits`、`d_in % group_size == 0`、scales と biases の形が
`[d_out, d_in / group_size]`。`bits ∈ {2, 3, 4, 5, 6, 8}`、`group_size ∈ {32, 64, 128}`
(libmlx.a のカーネルがこの範囲で実体化されている)。Phase 1 では `mode: "affine"` のみ。

### 3.4 読み込み: 2 つの経路

**(A) 量子化済み checkpoint を読む。** mlx-community の `*-4bit` のように weight / scales
/ biases が揃っている場合。宣言が 3 つ組で、ファイルにも 3 つあるので、`plan.rs` の
`Source::Pretrained` は今のまま通る。宣言 dtype(u32 / bf16)とファイルが一致するので
`as_dtype` も走らない。**engine の読み込み経路に変更は要らない。**

**(B) bf16 checkpoint を読み込み時に量子化する。** ファイルに `{name}.weight` が
bf16 `[d_out, d_in]` でだけある場合。`Source::Pretrained` に 1 つ規則を足す:

> 宣言が量子化の 3 つ組で、ファイルにその `.scales` が無く、`.weight` が宣言の
> 「元の形」`[d_out, d_in]` で浮動小数なら、`mlx_quantize` で 3 つを作る。
> それ以外の不一致は今まで通り断る。

「元の形」は `ParameterSpec` に書かずに済む: `[d_out, d_in × bits / 32]` と `bits` から
逆算できる。3 つ組のうちどれを見ているかに関わらず同じ量子化結果を使うため、
engine は `{name}` 単位で一度だけ量子化して 3 つに配る。

メモリのピークは「量子化済みの全体 + bf16 の 1 層分」。MLX は lazy なので、
層ごとに `eval` して bf16 の方を手放さないとピークが bf16 全体になる。Phase 2 の
テストで測る数字はこれである。

(A) を先にやる(Phase 1)。配布されている 4 bit を読めれば、LoRA の学習そのものは
始められるし、(B) の量子化の正しさは (A) のファイルと突き合わせて確かめられる
(同じ bf16 元から Torobi が量子化したものと mlx-lm が量子化したものが bit 一致するか)。

### 3.5 DSL

`adapting` と同じ形で、量子化をスコープに置く。

```ruby
quant   = Torobi::Quantization.new(bits: 4, group_size: 64, on: :linears)
adapter = Torobi::LoRA.new(rank: 8, alpha: 16, on: %w[q_proj v_proj])

Torobi.graph do |g|
  g.quantized(quant) do
    g.adapting(adapter) do
      ... the model ...
    end
  end
end
```

`linear` の分岐が 1 つ増える:

```ruby
def linear(x, d_out, name:, bias: true)
  d_in = concrete_last_dim!(x, ...)
  y = if quantization.wraps?(scoped(name))
        quantization.apply(self, x, d_in:, d_out:, name:)   # 3 param + quantized_matmul
      else
        w = param("#{name}.weight", [d_out, d_in], dtype: x.dtype, init: ...)
        matmul(x, w.transpose(axes: [1, 0]))
      end
  y += param("#{name}.bias", ...) if bias
  y += adapter.contribution(self, x, d_in:, d_out:, name:) if adapter.wraps?(scoped(name))
  y.named(name)
end
```

`Quantization#apply` は `param("#{name}.weight", [d_out, d_in * bits / 32], dtype: :u32,
trainable: false, init: ...)` と scales / biases(`x.dtype`、`trainable: false`)を宣言し、
`emit("quantized_matmul", inputs: [x, w, s, b], attrs: { group_size:, bits:, mode: })`
を返す。bias(`{name}.bias`)は量子化しない。mlx-lm も同じ。

`on:` の既定は「全 `linear`」である。mlx-lm の `*-4bit` は `lm_head` も量子化するが、
`embed_tokens` も量子化する(`QuantizedEmbedding`)。embedding は `take` であって
`matmul` ではないので、本計画では **embedding は bf16 のまま**とし、`on:` で除外する。
mlx-community の checkpoint を読むときは、`embed_tokens.scales` がファイルにあっても
宣言には無いので、読み込みが「ファイルにあって宣言に無い」を無視するか断るかに依る。
今の `plan.rs` は宣言側から引くので無視する。ただし tied embedding(`lm_head` が
`embed_tokens` を共有)のモデルでは `lm_head` も bf16 になるので、`on:` の既定から
`lm_head` を外す必要がある。これは Phase 1 のテストで実物(Qwen2.5-0.5B は tied)に
当てて決める。

**Adapter のスコープとの関係。** `quantized` が外、`adapting` が内。`param` の
`trainable &&= @adapter.trains?(path)` はそのまま効き、量子化パラメータは最初から
`trainable: false` なので二重に凍る。順序が逆でも結果は同じだが、「base をどう持つか」
を先に、「何を学習するか」を後に書く方が読める。

`initializer` について。量子化パラメータは checkpoint から来るもので、`fresh:` で
作るものではない。initializer は `zeros` を置くが、`fresh:` に量子化パラメータが
含まれていたら `GraphConfig` が断る(「量子化された重みは初期化できない。bf16 で
宣言して量子化するか、量子化済みのファイルを与える」)。

**モデル記述側**は `causal_lm(config, seq:, adapter:, dtype:, quantization:)` に 1 つ
増えるだけで、中は `quantized quantization do adapting adapter do ... end end`。
`config.json` に `quantization` があれば `Quantization.read(config)` で作れるようにする
(`description.rb` が `rope_scaling` を読むのと同じ場所)。

### 3.6 境界: fetch / put / checkpoint / export

| 操作 | 量子化パラメータに対して |
| --- | --- |
| `fetch(".weight")` | Phase 1: 断る(u32 は境界が運ばない)。Phase 3: **dequantize した値**を f32 で返す。「ラベルは payload が何であるかを言う」(§15.51)に従い、返るのはその重みが表す数であって packed ではない |
| `fetch(".scales")` など | bf16 なので今まで通り f32 で返る |
| `put` | 断る。量子化重みは置き換えるものではない(置き換えるなら bf16 で宣言し直す) |
| `freeze` / `thaw` | `thaw` が量子化パラメータに当たったら断る(trainable に戻せない) |
| checkpoint 保存 | 全パラメータを書くので u32 の packed もそのまま書かれる。manifest の dtype に `"u32"` が増える。再開は形と dtype の一致で通る |
| checkpoint 容量 | base 4.5 bit + adapter。32B で 17 GiB。「adapter だけを書く」は QLoRA 固有ではなく LoRA 全体の話なので、ここでは扱わない(§7) |
| `export_model!` | `refuse_adapted!` は残る。Phase 3 で 2 つ足す: (1) **adapter だけを PEFT 形式**(`adapter_model.safetensors` + `adapter_config.json`)で書く。QLoRA の成果物として実際に使うのはこれである。(2) merge: `dequantize(W) + scale × BA` を bf16 で書く。「量子化された base を書く」は mlx-lm 形式になり、別の話 |

### 3.7 プラットフォーム

| | Metal(macOS) | CUDA 12(Linux) |
| --- | --- | --- |
| `quantize` / `quantized_matmul` | MLX の最初からある | `affine_quantize` / `qmm_sm80` / `qmm_sm90` のカーネルが pin の `libmlx.a` にある(§2.3)。sm86(RTX 3080)は sm80 経路 |
| 状態 | 動くはず | **未実測**。Phase 0 で Linux の Rust テストを通す |
| bf16 × u32 の混在 | 問題なし | 問題なし(x は bf16、packed は u32、カーネルが読む) |

CPU backend(Linux CPU の MLX build)は pin に無い(plan.md §5)。本計画も扱わない。

## 4. 仕様

### 4.1 IR

- `IR::Dtype::ALL` に `:u32` を足す。`cast` の対象にはしない(`shape.rb` の `cast` で
  u32 を断る。数値を u32 に変換する用途は無い)。
- `config/ops.yml`:

  ```yaml
  - name: quantized_matmul
    inputs: [3, 4]            # x, weight, scales[, biases]
    attributes: { group_size: integer, bits: integer, mode: string }
    shape_rule: quantized_matmul
  ```
- `GraphConfig` の構築時検査に 2 つ: 「`quantized_matmul` の weight / scales / biases に
  当たる parameter が trainable である」「`fresh:` に u32 の parameter が含まれる」は
  `ConfigError`。

### 4.2 engine

- `engine/src/mlxc/ops.rs`: `quantize(w, group_size, bits, mode) -> (w, scales, biases)`、
  `dequantize(w, scales, biases, group_size, bits, mode, dtype)`、
  `quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode)`。
  `mlx_optional_int` は `{ value, has_value: true }` で渡す。`mode` は `CString`。
  `global_scale` は null(nvfp4 専用)。
- `engine/src/op.rs`: `Op::QuantizedMatmul { group_size: i32, bits: i32, mode: String }`。
- `engine/src/interp.rs`: `ins[0].quantized_matmul(&ins[1], &ins[2], ins.get(3), true, …)`。
- `engine/src/tensor.rs`: `dtype_spelling` / `dtype_named` に `"u32"`。
- `engine/src/plan.rs` `Source::Pretrained`(Phase 2): §3.4 (B) の規則。量子化は
  `{name}` 単位で一度。
- `engine/src/session.rs`: `fetch` が u32 を断る(Phase 1)→ dequantize して返す(Phase 3)。
  `put` と `set_frozen(…, false)` が u32 を断る。
- checkpoint: dtype 綴りに `u32` が増える以外は変更なし。

### 4.3 Ruby

- `Torobi::Quantization = Data.define(:bits, :group_size, :mode, :on)`。
  `wraps?(path)` は `LoRA` と同じ最後のセグメント照合。`on:` は配列か `:linears`。
  `Quantization.read(config)` が `config.json` の `quantization` を読む。
- `DSL::Builder#quantized(quant) { }`、`NoQuantization`(`adapting` / `NoAdapter` と同形)。
- `DSL::Layers#linear` の分岐(§3.5)。
- `Models::{Llama, Gemma3}.causal_lm(…, quantization: nil)`。ModernBERT は対象外
  (encoder は小さく、量子化の動機が無い)。
- `Export`: Phase 3 で `publish_adapter`(PEFT 形式)と `merge`。

### 4.4 エラーは宣言時に出す

| 状況 | どこで | メッセージが名指しするもの |
| --- | --- | --- |
| 量子化パラメータが trainable | `GraphConfig` 構築 | `g.quantized` の外に出すか、`LoRA` の `on:` を使う |
| `fresh:` に u32 | `GraphConfig` 構築 | bf16 で宣言して読み込み時量子化にするか、量子化済みファイルを与える |
| `d_in % group_size != 0` | `shape.rb` | その `linear` の名前と `d_in`、使える `group_size` |
| ファイルの weight が宣言の元の形でも量子化形でもない | `plan.rs` | 両方の形を書いて見せる |
| `thaw` が量子化パラメータに当たる | `session.rs` | 「量子化された重みは学習できない」 |

## 5. 計画

段取りは LoRA(§15.48-50)に倣う: engine → IR → DSL → 実物、各段でテストが
「何を言っているか」を表にして閉じる。数は現在 Ruby 453 件 / Rust 183 件。

### Phase 0: MLX の量子化演算を Rust から呼ぶ(IR に触れない)

やること: `engine/src/mlxc/ops.rs` に 3 つの束縛。Rust のテストだけ。

| テストが言うこと | 内容 |
| --- | --- |
| 往復の誤差が量子化の理論値に収まる | `dequantize(quantize(W))` と `W` の差。affine 4 bit g64 なら group 内の最大値の 1/15 以下 |
| `quantized_matmul` は dequantize してから matmul したものと一致する | bf16 で 1e-2 相対、f32 で 1e-5 |
| x についての勾配が取れる | `value_and_grad` で `sum(quantized_matmul(x, …))` を x で微分 |
| **weight についての勾配は MLX が断る** | 例外になることをテストで固定する(§3.2 の根拠) |
| **Linux CUDA で同じ 4 つが通る** | bear1(RADV は MLX に無いので対象外)ではなく RTX 3080 機で。CI の ubuntu は bind + link まで |

出口: 上の 5 行。Linux で通らなければ、計画はここで止めて原因を plan.md §15 に書く。

### Phase 1: 量子化済み checkpoint を読んで LoRA を当てる

やること: `u32` dtype、`quantized_matmul` op(ops.yml / shape.rb / op.rs / interp.rs)、
`GraphConfig` の 2 つの検査、`Torobi::Quantization`、`g.quantized`、`linear` の分岐、
`causal_lm(quantization:)`、`fetch` / `put` / `thaw` の拒否。

実物: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`(weight / scales / biases、
`quantization: {group_size: 64, bits: 4}`)。既存の LoRA テストが Qwen2.5-0.5B の bf16 を
使っているので、同じモデルの 4 bit 版である。

| テストが言うこと | 内容 |
| --- | --- |
| 同じ記述、1/3.5 のバイト数 | 公開 config に対して 0.92 GiB → 0.26 GiB |
| **forward が mlx-lm と一致する** | 同じ 4 bit ファイル、同じ入力で logits を比べる。bf16 で 1e-2 |
| adapter だけが微分される | 宣言でも、開いた run の `trainable` でも。量子化パラメータは 3 つとも入らない |
| 学習は adapter を動かし、base は 1 バイトも動かない | packed u32 と scales / biases を step 前後で比べる |
| 量子化パラメータを trainable にした宣言は構築時に断られる | step ではなく `GraphConfig` で |
| checkpoint を書いて読み戻せる | manifest に `"u32"` が現れ、再開した run が同じ loss を出す |
| tied embedding のとき `lm_head` がどうなるか | 実物で決めて、テストで固定する(§3.5) |

出口: 上の 7 行と、Linux で同じスイートが通ること。

### Phase 2: bf16 checkpoint を読み込み時に量子化する

やること: `plan.rs` `Source::Pretrained` の規則(§3.4 (B))、層ごとの `eval`。

| テストが言うこと | 内容 |
| --- | --- |
| **Torobi の量子化と mlx-lm の量子化が bit 一致する** | `Qwen2.5-0.5B-Instruct`(bf16)を Torobi で読み込み量子化したものと `mlx-community/…-4bit` の weight / scales / biases を比べる。同じ MLX の `quantize` なので一致するはず。しなければどちらかの入力が違う |
| ピークメモリが「量子化済み全体 + 1 層 bf16」に収まる | `Torobi::Memory` で測る。bf16 全体より小さいことを数字で |
| 宣言とファイルの形がどちらにも合わなければ断る | メッセージが両方の形を書く |

出口: 上の 3 行。ここで 7B 級を RTX 3080 で、32B 級を Mac 48 GB で開いて、§1.2 の
表の「見込み」を実測に置き換える。

### Phase 3: 出口を作る(fetch / export)

| 追加 | 内容 |
| --- | --- |
| `fetch` が dequantize した値を返す | §3.6。f32 で、その重みが表す数 |
| `export_adapter!`(PEFT 形式) | `adapter_model.safetensors`(`base_model.model.{path}.lora_A.weight` の綴り)と `adapter_config.json`(`r`, `lora_alpha`, `target_modules`)。PEFT / mlx-lm で読めることをテスト |
| merge | `dequantize(W) + (alpha / rank) × BA` を bf16 で書く。`refuse_adapted!` はこれが通る経路を得る。量子化 base のまま書くのは mlx-lm 形式であり、ここではやらない |

出口: PEFT で adapter を読み込んで、Torobi と同じ logits が出ること。

### Phase 4(据え置き): §7 に挙げたもの

## 6. リスク

| リスク | 備考 | 手当て |
| --- | --- | --- |
| CUDA の量子化カーネルが sm86 で遅い・合わない | シンボルはあるが実測ゼロ | Phase 0 で止める判断を先に置く |
| `mlx_optional_int` を bindgen 経由で値渡しする ABI | 小さな struct の値渡し。engine はまだ `mlx_optional_*` を 1 つも渡していない(`grep mlx_optional engine/src/mlxc` が空)ので、これが最初になる | Phase 0 で判明する。bindgen は struct を値で渡せるので問題は無い見込み |
| 読み込み時量子化のメモリピーク | lazy eval で bf16 全体が残る | 層ごと `eval`。Phase 2 で測る |
| mlx-community の checkpoint は何を量子化しているかがモデルごとに違う | embedding / lm_head / 一部の層を外しているものがある | `on:` を既定に頼らず、ファイルの `.scales` の有無から `Quantization.read` が決められるようにする(Phase 1 の tied embedding と同じ場所) |
| `d_in` が `group_size` で割り切れないモデル | Gemma3 の head_dim 256 系は大丈夫。小さいモデルの一部 | `shape.rb` で宣言時に断る。`group_size: 32` に落とす選択肢を示す |
| QLoRA の品質(4 bit base での学習が bf16 LoRA に劣るか) | 原論文は NF4 + double quant。MLX の affine 4 bit は別物 | Torobi の範囲外。llm-bench 側で bf16 LoRA と比較する(Phase 1 の 0.5B で loss 曲線を並べるのは安い) |
| `quantized_matmul` の VJP が x についてしか無いこと | 将来 MLX が w の勾配を足しても、量子化重みを学習する動機は無い | `GraphConfig` で断る側に倒す |

## 7. やらないこと

- **MoE**(`gather_qmm`)。llm-jp-4.1-32b-a3b は MoE で、Torobi には MoE の記述も
  router もまだ無い。量子化以前の話なので別の計画にする。
- **embedding の量子化**(`QuantizedEmbedding`)。`take` の量子化版で、メモリの節約は
  vocab × dim 分(0.5B で 0.26 GiB の 1/4 を占めるので小さくはない)。Phase 4。
- **`mxfp4` / `mxfp8` / `nvfp4`**。CUDA 側に `CublasQQMM` があるが、Metal 側の対応と
  checkpoint の配布形式がまだ `affine` 中心。mode は属性なので、足すときは `shape.rb`
  の検査と Phase 0 のテストを増やすだけで済む。
- **NF4 / double quantization**(原論文の形式)。MLX に無い。
- **adapter だけの checkpoint**。LoRA 全体の話で、QLoRA 固有ではない。
- **量子化 base を mlx-lm 形式で export する**。読む側(mlx-lm)の形式に合わせる話で、
  Torobi の学習結果の出口としては adapter(PEFT)と merge(bf16)で足りる。
- **CPU backend**、**Vulkan**。MLX に無い。

## 8. 付記: この計画を書くために確かめたこと

- `engine/src/mlxc/` に量子化の束縛が無いこと(`grep mlx_quantize` が空)。
- mlx-c `ops.h` の 4 関数のシグネチャ(§2.3 に転記)。
- pin の `libmlx.a`(CUDA 12)に `affine_quantize` / `qmm_sm80` / `qmm_sm90` /
  `QuantizedMatmul::eval_cpu` が入っていること。GPU 実装の有無はシンボルから読んだ
  もので、実行はしていない。
- `plan.rs:469-503` が `shaped()`(形の一致)→ `as_dtype`(宣言 dtype へ)の順で、
  形の不一致は断ること。(B) の規則はこの手前に入る。
- `export.rb` の `refuse_adapted!` が merge 未実装を理由に断っていること。
- `linear` が adapter の寄与を bias の後に足し、`contribution` が `x.dtype` で A, B を
  作ること。量子化しても adapter 側は変わらない根拠。
