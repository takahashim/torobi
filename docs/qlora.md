# Torobi QLoRA 対応 仕様と計画

- Status: **Phase 0〜3 を実装。Rust のテストは GPU で全部通過、実物の 4 bit ファイルは
  inventory と mlx-lm との forward の比較まで通過**(2026-11-08、plan.md §15.76)。残るのは
  7B / 32B の実測と PEFT での読み戻し(§9)
- Date: 2026-11-08(同日、コードと突き合わせたレビューを反映。§8)
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

Phase 1 は「engine の読み込み経路に変更は要らない」では済まない。mlx-lm は embedding も
量子化して配るので、embedding を bf16 で持つ Torobi がそのファイルを読むには、読み込み時に
**dequantize する規則**が 1 つ要る(§3.4 (A'))。dequantize の束縛は Phase 0 で作るので
足す量は小さい。

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

本計画は embedding を量子化しない(§3.5、§7)ので、実際に持つのは「linear は 4.5 bit、
embedding は bf16」である。小さいモデルほど embedding の比率が大きく、0.5B では
パラメータの 28 % を占めるので、全部 4.5 bit にした数字とは別に書く。

| モデル | パラメータ | bf16 | 全部 4.5 bit | **linear 4.5 bit + embedding bf16** | 備考 |
| --- | ---: | ---: | ---: | ---: | --- |
| Qwen2.5-0.5B | 0.49 B | 0.92 GiB | 0.26 GiB | **0.44 GiB**(1/2.1) | テスト用。既存の LoRA テストと同じ実物。embedding 136 M が 272 MB、linear 358 M が 201 MB |
| Qwen2.5-7B / Llama-3.1-8B | 7.6 / 8.0 B | 14.2 / 15.0 GiB | 4.0 / 4.2 GiB | 4.7 / 4.9 GiB | RTX 3080 16 GB で活性と adapter を含めて載る見込み |
| Qwen2.5-32B | 32.8 B | 61 GiB | 17.2 GiB | **18.2 GiB** | Mac 48 GB で載る。16 GB NVIDIA には載らない |
| llm-jp-4.1-33b | 33 B | 61 GiB | 17.3 GiB | 17.3 GiB + embedding(vocab 次第で 1 GiB 前後) | 同上 |
| llm-jp-4.1-32b-a3b(MoE) | 32 B | 60 GiB | 17 GiB | — | **対象外**。expert の `gather_qmm` と MoE の記述が別に要る(§7) |

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
| dtype 語彙 | `lib/torobi/ir/dtype.rb` `ALL = %i[f32 bf16 i32 bool]`、`engine/src/tensor.rs` | engine の `Dtype` enum(`engine/src/mlxc/array.rs:103`)には `Uint32` が既にある。IR 語彙に無いだけ |
| 読み込みと dtype 変換 | `engine/src/plan.rs:469-503` | `Source::Pretrained` はファイルの配列を `shaped()`(形が一致することを要求)で受け、宣言 dtype と違えば `as_dtype`。ファイルの配列は `files: HashMap<String, Array>` に lazy なハンドルとして握られ、`materialize(&self)` → `eval(params)` の間ずっと生きている(`plan.rs:143-147`、`:373`) |
| 学習状態 | `engine/src/state.rs` | `params: Vec<Array>` に全パラメータ、`argnums` に微分する位置。凍結は argnums の出し入れ |
| checkpoint | `engine/src/checkpoint.rs` | `parameters.safetensors` に全パラメータを書く。manifest に path / shape / dtype / trained |
| export | `lib/torobi/export.rb` `refuse_adapted!`、`engine/src/export.rs:98-101` | adapter つきの run は書かずに断る。merge は未実装(「checkpoint が両方を持っているので失われない」)。engine 側は**全パラメータを `as_dtype(Float32)` して書く**。adapter 無しの run は `refuse_adapted!` を通る |
| `param_norm` | `engine/src/state.rs:220` | `self.params` **全体**の global norm。trainable に限っていない |
| op 語彙 | `config/ops.yml`(単一の真実)→ `lib/torobi/shape.rb`(形規則)、`engine/src/op.rs`(enum と parse)、`engine/src/interp.rs`(実行) | 属性つき op の前例: `cast` の `dtype`、`sdpa` の `scale` / `causal` |
| 勾配 | `engine/src/mlxc/transforms.rs` | `mlx_value_and_grad`。backward は MLX の VJP。手書きの backward は無い |

### 2.2 ない

- 量子化演算の束縛。`engine/src/mlxc/ops.rs` の `unary!` / `binary!` マクロ群に
  `mlx_quantize` 系は無い。`mlx_optional_int`(`{int value; bool has_value}`)と
  `const char* mode` を渡す束縛もまだ無い(`CString` の前例は `io.rs` にある)。
- IR 語彙に `u32`。packed weight は `uint32` の `[d_out, d_in × bits / 32]`。
- 1 つの重みが 3 つの配列(weight / scales / biases)になる、という形のパラメータ。
- `fetch` / `put` の境界は i32 と f32 の 2 つしか運ばない(§15.51)。u32 は運べない。
- 「宣言は浮動小数、ファイルは量子化済み」を読む経路。`shaped()` は形の一致しか見ないので、
  packed `[d_out, d_in × bits / 32]` を `[d_out, d_in]` の宣言に当てると断る。

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

- `mlx::core::cu::affine_quantize<__nv_bfloat16|__half, 32|64|128, 2..8>` — 量子化の GPU
  カーネル(`nm -C` で確認、`affine_dequantize` も同じ組)。最初の写しに g64 が無かったのは
  省略で、実体はある
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
**embedding も同じ形で量子化されている**(`nn.quantize` は `nn.Embedding` を
`QuantizedEmbedding` にする)。`model.embed_tokens.weight` は u32 の packed で、bf16 の表は
ファイルに無い。tied のモデルでは `lm_head` も無く、mlx-lm は量子化された表で
`quantized_matmul` を取る。
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

勾配について。MLX の `QuantizedMatmul::vjp` は **packed weight については例外を投げる**
(「no gradient wrt the quantized weights」)。**scales と biases については勾配を返す**
(2024 年末に入った変更。Phase 0 のテストが GPU で通り、確定した)。つまり MLX が断るのは weight だけで、
scales / biases を学習しないのは Torobi の方針である: QLoRA が学習するのは adapter であって
base の量子化パラメータではなく、それを動かした run は「量子化 base に LoRA を当てた」とは
言えなくなる。Torobi では 3 つとも `trainable: false` で宣言されるので argnums に入らない。

DSL 経由で trainable にする道は無い。`param` の `trainable &&= @adapter.trains?(path)` は
下げる一方で、`Quantization#apply` は `trainable: false` を渡す。残るのは手書きの IR
(JSON)だけで、そこは **`GraphConfig` が構築時に「u32 の parameter が trainable」を断る**
(loss が f32 でないときと同じ扱い、§15.51)。scales / biases まで node の入力を辿って
検査するのはやらない: 入力が変換を挟むと外れるし、DSL では起こり得ない。

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

**(A) 量子化済み checkpoint の、量子化された linear を読む。** mlx-community の `*-4bit`
のように weight / scales / biases が揃っていて、宣言も 3 つ組の場合。ファイルにも 3 つ
あるので、`plan.rs` の `Source::Pretrained` は今のまま通る。宣言 dtype(u32 / bf16)と
ファイルが一致するので `as_dtype` も走らない。

**(A') 量子化済み checkpoint の、Torobi が bf16 で持つものを読む。** embedding(§2.4)と、
`on:` で量子化から外した linear がこれに当たる。宣言は浮動小数 `[d_out, d_in]`、ファイルは
u32 の `.weight` `[d_out, d_in × bits / 32]` と `.scales`(と `.biases`)。`shaped()` は
形で断るので、その手前に 1 つ規則を足す:

> 宣言が浮動小数で、ファイルの `.weight` が u32、同じ名前の `.scales` があれば、
> `mlx_dequantize` して宣言 dtype にする。`group_size` / `bits` は**ファイルの形から
> 逆算する**: packed `[d_out, d_in × bits / 32]` と宣言 `[d_out, d_in]` で bits が、
> scales の `[d_out, d_in / group_size]` で group_size が決まる。mode は affine
> (biases を持つ唯一の mode で、mlx-lm が配るのもこれ)。逆算が §3.3 の範囲に
> 収まらなければ、両方の形を書いて断る。

これが無いと、選んだ実物(Qwen2.5-0.5B、tied)は Phase 1 で開けない。dequantize の束縛は
Phase 0 で作るので、足すのは規則 1 つである。**Phase 1 の engine 側はこの規則が本体**になる。

**(B) bf16 checkpoint を読み込み時に量子化する。** ファイルに `{name}.weight` が
bf16 `[d_out, d_in]` でだけある場合。`Source::Pretrained` に 1 つ規則を足す:

> 宣言が量子化の 3 つ組で、ファイルにその `.scales` が無く、`.weight` が宣言の
> 「元の形」`[d_out, d_in]` で浮動小数なら、`mlx_quantize` で 3 つを作る。
> それ以外の不一致は今まで通り断る。

「元の形」は `ParameterSpec` に書かずに済む: `[d_out, d_in × bits / 32]` と `bits` から
逆算できる。3 つ組のうちどれを見ているかに関わらず同じ量子化結果を使うため、
engine は `{name}` 単位で一度だけ量子化して 3 つに配る。

メモリのピークを「量子化済みの全体 + bf16 の数層分」に収めるには、**ファイルの配列の
ハンドルを手放す**必要がある。原因は eval の粒度ではない。`Source::Pretrained.files` が
`load_safetensors` の lazy な配列を `HashMap` に握ったまま `materialize(&self)` が走り、
`eval(params)` が終わるまで `source` が生きている(§2.1)。eval で Load が実体化した bf16
バッファは、map が握っている限り解放されない。層ごとに eval しても同じである。
MLX 自身は eval の中で消費し終えた入力を detach して解放するので、ハンドルさえ手放せば
1 回の eval でもピークは収まるはずである。

手当ては `materialize` を `self` を消費する形(か `&mut self`)にして、量子化に使った配列を
map から `remove` してから eval すること。これは (A') にも効く(dequantize の元の packed を
握り続けない)。Phase 2 のテストで測る数字はこれで、「bf16 全体より小さい」を数字で言う。

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

`on:` の既定は「全 `linear`」である。embedding は `take` であって `matmul` ではないので、
本計画では **embedding は bf16 のまま**とする(`Quantization` は `linear` にしか効かないので、
除外を書く必要は無い)。mlx-community の checkpoint は embedding も量子化しているので
(§2.4)、読むときは §3.4 (A') で dequantize する。`embed_tokens.scales` /
`.biases` は dequantize に使われ、それ以外の「ファイルにあって宣言に無い」配列は、今の
`plan.rs` が宣言側から引くので無視される。

tied embedding(Qwen2.5-0.5B がそう)では `lm_head` は `linear` ではなく、dequantize した
表を転置した bf16 の `matmul` になる(`llama.rb` の `head`)。mlx-lm は量子化された表で
`quantized_matmul` を取るが、dequantize した値は同じなので、forward の差は bf16 の丸めの
範囲に収まる(Phase 1 のテストで確かめる)。untied のモデルでは `lm_head` は `linear` で、
`on:` の既定に含まれる。

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
| `fetch(".weight")` | **dequantize した値**を f32 で返す(Phase 1 から)。「ラベルは payload が何であるかを言う」(§15.51)に従い、返るのはその重みが表す数であって packed ではない。Phase 1 の「base は 1 バイトも動かない」のテストがこれを使うので、断るだけの段階は置かない |
| `fetch(".scales")` など | bf16 なので今まで通り f32 で返る |
| `put` | 断る。量子化重みは置き換えるものではない(置き換えるなら bf16 で宣言し直す) |
| `freeze` / `thaw` | 届かない。`set_frozen` は宣言 trainable の集合の中だけを動かす(§4.4) |
| `param_norm` | `self.params` 全体のノルム(§2.1)に packed u32 が混ざる。**u32 を除く**(ノルムは「重みの大きさ」で、packed はそれではない) |
| checkpoint 保存 | 全パラメータを書くので u32 の packed もそのまま書かれる。manifest の dtype に `"u32"` が増える。再開は形と dtype の一致で通る |
| checkpoint 容量 | base 4.5 bit + adapter。32B で 17 GiB。「adapter だけを書く」は QLoRA 固有ではなく LoRA 全体の話なので、ここでは扱わない(§7) |
| `export_model!` | packed の重みは **dequantize した数を f32 で**書き、scales / biases は書かない(Phase 3 で。Phase 1 の間は u32 を含む export を断っていた)。`refuse_adapted!` は `merge:` 無しのとき残る。Phase 3 で 2 つ足した: (1) **adapter だけを PEFT 形式**(`adapter_model.safetensors` + `adapter_config.json`)で書く。QLoRA の成果物として実際に使うのはこれである。(2) merge: `dequantize(W) + scale × BA` を bf16 で書く。「量子化された base を書く」は mlx-lm 形式になり、別の話 |

### 3.7 プラットフォーム

| | Metal(macOS) | CUDA 12(Linux) |
| --- | --- | --- |
| `quantize` / `quantized_matmul` | MLX の最初からある | `affine_quantize` / `qmm_sm80` / `qmm_sm90` のカーネルが pin の `libmlx.a` にある(§2.3)。sm86(RTX 3080)は sm80 経路 |
| 状態 | 動くはず | **未実測**。Phase 0 で Linux の Rust テストを通す |
| bf16 × u32 の混在 | 問題なし | 問題なし(x は bf16、packed は u32、カーネルが読む) |

CPU backend(Linux CPU の MLX build)は pin に無い(plan.md §5)。本計画も扱わない。

## 4. 仕様

### 4.1 IR

- `IR::Dtype::ALL` に `:u32` を足す。`cast` の対象にはしない。`shape.rb` の `cast` は
  `IR::Dtype.check!` で `ALL` を見るので(`shape.rb:109`)、`ALL` に足した時点で通ってしまう。
  **明示の除外**が要る(数値を u32 に変換する用途は無い)。
- `config/ops.yml`:

  ```yaml
  - name: quantized_matmul
    inputs: [3, 4]            # x, weight, scales[, biases]
    attributes: { group_size: integer, bits: integer, mode: string }
    shape_rule: quantized_matmul
  ```
- `GraphConfig` の構築時検査に 2 つ: 「u32 の parameter が trainable である」「`fresh:` に
  u32 の parameter が含まれる」は `ConfigError`。dtype だけで判る検査にする(§3.2)。
- 読み込み時の dequantize(§3.4 (A'))はファイルの形から `group_size` / `bits` を
  逆算するので、graph に書くものは無い。`fetch` の dequantize は、`Plan` が
  `quantized_matmul` の入力を parameter node まで辿って作る表(packed → scales /
  biases / quantization)を使う。量子化(§3.4 (B)、Phase 2)は宣言側の u32 と
  `quantized_matmul` の属性が目標を言う。

### 4.2 engine

- `engine/src/mlxc/ops.rs`: `quantize(w, group_size, bits, mode) -> (w, scales, biases)`、
  `dequantize(w, scales, biases, group_size, bits, mode, dtype)`、
  `quantized_matmul(x, w, scales, biases, transpose, group_size, bits, mode)`。
  `mlx_optional_int` は `{ value, has_value: true }` で渡す。`mode` は `CString`。
  `global_scale` は null(nvfp4 専用)。
- `engine/src/op.rs`: `Op::QuantizedMatmul { group_size: i32, bits: i32, mode: String }`。
- `engine/src/interp.rs`: `ins[0].quantized_matmul(&ins[1], &ins[2], ins.get(3), true, …)`。
- `engine/src/tensor.rs`: `dtype_spelling` / `dtype_named` に `"u32"`。
- `engine/src/plan.rs` `Source::Pretrained`(Phase 1): §3.4 (A') の規則(dequantize して
  宣言 dtype に)。(Phase 2): §3.4 (B) の規則。量子化は `{name}` 単位で一度。どちらも
  `materialize` がファイルの配列を map から取り出してから eval する形に直す(§3.4)。
- `engine/src/state.rs`: `fetch` が u32 を dequantize して f32 で返す(Phase 1)。
  `put` が u32 を断る。`param_norm` が u32 を除く。`set_frozen` は変えない(§4.4)。
- `engine/src/export.rs`: u32 を含むモデルは断る(Phase 1)。Phase 3 で merge の経路を得る。
- checkpoint: dtype 綴りに `u32` が増える以外は変更なし。

### 4.3 Ruby

- `Torobi::Quantization = Data.define(:bits, :group_size, :mode, :on)`。
  `wraps?(path)` は `LoRA` と同じ最後のセグメント照合。`on:` は配列か `:linears`。
  `Quantization.from_config(raw)` が `config.json` の `quantization` から `bits` /
  `group_size` を、`from_header(raw, header)` が **safetensors のヘッダ(JSON)から
  `.scales` を持つモジュールの一覧**を読んで `on:` を決める。`from_directory(dir)` は
  両方を checkpoint のディレクトリから。`config.json` だけでは何が量子化されているか
  判らない(§6)。
- `DSL::Builder#quantized(quant) { }`、`NoQuantization`(`adapting` / `NoAdapter` と同形)。
- `DSL::Layers#linear` の分岐(§3.5)。
- `Models::{Llama, Gemma3}.causal_lm(…, quantization: nil)`。ModernBERT は対象外
  (encoder は小さく、量子化の動機が無い)。
- `Export`: Phase 3 で `publish_adapter`(PEFT 形式)と `merge`。

### 4.4 エラーは宣言時に出す

| 状況 | どこで | メッセージが名指しするもの |
| --- | --- | --- |
| u32 の parameter が trainable | `GraphConfig` 構築 | `g.quantized` の外に出すか、`LoRA` の `on:` を使う |
| `fresh:` に u32 | `GraphConfig` 構築 | bf16 で宣言して読み込み時量子化にするか、量子化済みファイルを与える |
| `d_in % group_size != 0` | `shape.rb` | その `linear` の名前と `d_in`、使える `group_size` |
| ファイルの weight が宣言の元の形でも量子化形でもない | `plan.rs` | 両方の形を書いて見せる |
| 宣言が浮動小数、ファイルが packed なのに `.scales` が無い | `plan.rs` | 「量子化された weight だが scales が無い」と、ファイルにある名前を並べる |
| u32 を含むモデルの `export_model!` | `export.rs` | 「量子化された base は書けない」。Phase 3 の merge を指す |
| `fresh:` が u32 を名指す | `Weights#check_fresh!`(`Session.open`) | 「量子化された重みは初期化できない。量子化済みのファイルを与えるか、bf16 で宣言する」 |
| `thaw` が量子化パラメータに当たる | (起こらない) | `set_frozen` は宣言 trainable の集合(`candidates`)の中だけを動かし、量子化パラメータは `trainable: false` なので最初から外にある。u32 だけに当たる pattern は「no parameter matches」になる |

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
| **scales / biases についての勾配は MLX が返す**(見込み) | 返るならそれを固定し、「Torobi が断るのは方針」と §3.2 に書く。返らなければ §3.2 を直す |
| **Linux CUDA で同じ 6 つが通る、g64 で** | bear1(RADV は MLX に無いので対象外)ではなく RTX 3080 機で。`group_size` は 32 / 64 / 128 の 3 つとも(§2.3 のシンボルに 64 が見えない)。CI の ubuntu は bind + link まで |

出口: 上の 7 行。Linux で通らなければ、計画はここで止めて原因を plan.md §15 に書く。

**結果(2026-11-08)**: `engine/src/mlxc/quantized.rs`。7 行のうち 6 行をテストに書いた。
g64 の CUDA カーネルは `nm` で確認した(§2.3)。**どの行もまだ GPU で走らせていない**
(§9)。

### Phase 1: 量子化済み checkpoint を読んで LoRA を当てる

やること: `u32` dtype、`quantized_matmul` op(ops.yml / shape.rb / op.rs / interp.rs)、
`GraphConfig` の 2 つの検査、`Torobi::Quantization`(ヘッダから `on:` を読む)、`g.quantized`、
`linear` の分岐、`causal_lm(quantization:)`、`plan.rs` の (A') 規則と `materialize` の
ハンドルの手放し、`fetch` の dequantize、`put` / `thaw` / `export` の拒否、`param_norm` の除外。

実物: `mlx-community/Qwen2.5-0.5B-Instruct-4bit`(weight / scales / biases、
`quantization: {group_size: 64, bits: 4}`)。既存の LoRA テストが Qwen2.5-0.5B の bf16 を
使っているので、同じモデルの 4 bit 版である。

| テストが言うこと | 内容 |
| --- | --- |
| 同じ記述、1/2.1 のバイト数 | 公開 config に対して 0.92 GiB → **0.44 GiB**(embedding は bf16。§1.2)。`bf16_test.rb` と同じ数え方 |
| **量子化された embedding を読んで bf16 の表になる** | (A') の規則。`embed_tokens.weight` が u32 のファイルから、宣言どおり bf16 `[vocab, dim]` が出る。tied なので `lm_head` もこの表で動く(§3.5) |
| **forward が mlx-lm と一致する** | 同じ 4 bit ファイル、同じ入力で logits を比べる。oracle は `tools/qlora_reference.py`、`test/oracle/` に JSON。**実測**: 次のトークン一致、logits 8.2e-3、hidden 3.3e-2 / 5.9e-2(bf16 の揺れ。§9)。配線は f32 の packed 対 数値読みで 1.5e-4 |
| adapter だけが微分される | 宣言でも、開いた run の `trainable` でも。量子化パラメータは 3 つとも入らない |
| 学習は adapter を動かし、base は 1 バイトも動かない | `fetch` が返す dequantize 値と scales / biases を step 前後で比べる。packed そのものは checkpoint を書いて safetensors のバイトで比べる |
| u32 の parameter を trainable にした宣言は構築時に断られる | 手書き IR で。step ではなく `GraphConfig` で |
| checkpoint を書いて読み戻せる | manifest に `"u32"` が現れ、再開した run が同じ loss を出す |
| u32 を含むモデルの export は断られる | adapter 無しで開いた run で。packed が f32 として書かれないこと |
| 読み込み後にファイルの配列が残っていない | `Torobi::Memory` で、開いた直後の常駐が 0.44 GiB 側であって bf16 の表 + packed の合計ではないこと |

出口: 上の 9 行と、Linux で同じスイートが通ること。

**結果(2026-11-08)**: 9 行のうち、engine 無しで言えるもの(バイト数、宣言の形、構築時の
拒否、`cast`、ヘッダの読み方)は `test/quantization_test.rb` で通っている。engine が要る
もの((A') の読み込み、fetch の dequantize、step で base が動かないこと、export の拒否)は
`plan.rs` / `state.rs` / `export.rs` の Rust テストに書いたが、まだ走らせていない(§9)。
forward の mlx-lm との比較と、実物の inventory との突き合わせ
(`test/oracle/qwen2.5-0.5b-instruct-4bit.json`、無ければ skip)は、Hub に届く機械で
inventory を生成してから。

### Phase 2: bf16 checkpoint を読み込み時に量子化する

やること: `plan.rs` `Source::Pretrained` の規則(§3.4 (B))。ハンドルの手放しは Phase 1 で
済んでいる。

| テストが言うこと | 内容 |
| --- | --- |
| **Torobi の量子化と、pin した MLX と同じ版の mlx-lm の量子化が bit 一致する** | `Qwen2.5-0.5B-Instruct`(bf16)を Torobi で読み込み量子化したものと、**手元で同じ版の MLX の `mx.quantize` にかけたもの**を比べる。mlx-community のファイルは古い MLX で量子化されていて、affine の scale / bias の計算は版で変わっているので、配布物との bit 一致は求めない。配布物とは forward の 1e-2 で比べる |
| ピークメモリが「量子化済み全体 + 数層分の bf16」に収まる | `Torobi::Memory` で測る。bf16 全体より小さいことを数字で。§3.4 のハンドルの手放しが効いている証拠 |
| 宣言とファイルの形がどちらにも合わなければ断る | メッセージが両方の形を書く |

出口: 上の 3 行。ここで 7B 級を RTX 3080 で、32B 級を Mac 48 GB で開いて、§1.2 の
表の「見込み」を実測に置き換える。

**結果(2026-11-08)**: `plan.rs` の `quantized_on_read`。宣言が `quantized_matmul` の
3 つ組の 1 つで、ファイルに `{module}.weight` が数値であり `.scales` が無ければ、
`quantize` を module ごとに 1 回呼んで 3 つの宣言に配る(`Source::Pretrained` の
`quantized` キャッシュ。source ごと eval の前に捨てる)。目標の量子化は `Plan.packed`
(その matmul の属性)から取る。テストは「ファイルの bf16 を同じ MLX の `quantize` に
かけたものと bit 一致」「形が packed でも元の形でもなければ両方を書いて断る」
「開いた直後の常駐が bf16 全体より小さい」の 3 本(Rust、未実行。§9)。7B / 32B の
実測はまだ。

### Phase 3: 出口を作る(fetch / export)

| 追加 | 内容 |
| --- | --- |
| `export_adapter!`(PEFT 形式) | `adapter_model.safetensors`(`base_model.model.{path}.lora_A.weight` の綴り)と `adapter_config.json`(`r`, `lora_alpha`, `target_modules`)。PEFT / mlx-lm で読めることをテスト |
| merge | `dequantize(W) + (alpha / rank) × BA` を bf16 で書く。`refuse_adapted!` と Phase 1 の u32 拒否は、これが通る経路を得る。量子化 base のまま書くのは mlx-lm 形式であり、ここではやらない |

出口: PEFT で adapter を読み込んで、Torobi と同じ logits が出ること。

**結果(2026-11-08)**: `export.rs` に `write_adapter`(`adapter_model.safetensors`、
`base_model.model.{path}` の綴り、f32)と、`write` の merge(`merge: Option<f32>` が
adapter の scale。`W + scale × B A` を f32 で計算し、A / B と packed の scales / biases は
書かない)。packed の重みは adapter の有無に関わらず dequantize して書く(Phase 1 の
「u32 を含む export は断る」はこれで解けた)。Ruby は `export_model!(dir, merge: adapter)` と
`export_adapter!(dir, adapter:, base:, task_type:)`。`adapter_config.json` は `Export.adapter_config`
が PEFT の `LoraConfig` の形で書く(`r`, `lora_alpha`, `target_modules`, `peft_type: "LORA"`)。
merge した重みは **f32 で書く**(既存の export と同じ契約。§3.6 の「bf16 で書く」から
変えた。bf16 で欲しければ読む側で落とせる)。**PEFT で実際に読み戻す確認は未**(§9)。

### Phase 4(据え置き): §7 に挙げたもの

## 6. リスク

| リスク | 備考 | 手当て |
| --- | --- | --- |
| CUDA の量子化カーネルが sm86 で遅い・合わない | シンボルはあるが実測ゼロ | Phase 0 で止める判断を先に置く |
| `mlx_optional_int` を bindgen 経由で値渡しする ABI | 小さな struct の値渡し。engine はまだ `mlx_optional_*` を 1 つも渡していない(`grep mlx_optional engine/src/mlxc` が空)ので、これが最初になる | Phase 0 で判明する。bindgen は struct を値で渡せるので問題は無い見込み |
| 読み込み時のメモリピーク | `Source::Pretrained.files` がファイルの配列を握ったまま eval する(§3.4)。(A') の dequantize でも (B) の量子化でも、元の配列が残れば合計になる | `materialize` がハンドルを手放してから eval する。Phase 1 で入れ、Phase 1 / 2 の両方で測る |
| mlx-community の checkpoint は何を量子化しているかがモデルごとに違う | embedding / lm_head / 一部の層を外しているものがある。混合 bit のものもある | `Quantization.read(config, file:)` が safetensors のヘッダから `.scales` の有無を読んで `on:` を決める(§4.3)。混合 bit は Phase 1 では断る |
| CUDA の `affine_quantize` に g64 が無い | §2.3 のシンボル一覧に 32 と 128 しか見えない | Phase 0 の Linux の行で g64 を名指しで確かめる。無ければ (B) は CUDA で g32 / g128 に限る(読む方 (A) / (A') は `dequantize` と `quantized_matmul` のカーネル次第) |
| 配布物との量子化の bit 一致 | mlx-community のファイルは古い MLX で量子化されている | 比較対象を手元の同じ版の MLX にする(Phase 2) |
| `d_in` が `group_size` で割り切れないモデル | Gemma3 の head_dim 256 系は大丈夫。小さいモデルの一部 | `shape.rb` で宣言時に断る。`group_size: 32` に落とす選択肢を示す |
| QLoRA の品質(4 bit base での学習が bf16 LoRA に劣るか) | 原論文は NF4 + double quant。MLX の affine 4 bit は別物 | Torobi の範囲外。llm-bench 側で bf16 LoRA と比較する(Phase 1 の 0.5B で loss 曲線を並べるのは安い) |
| `quantized_matmul` の VJP が weight について無いこと | scales / biases は MLX が微分できる(見込み)。将来 MLX が w の勾配を足しても、量子化重みを学習する動機は無い | `GraphConfig` で断る側に倒す。scales / biases を断るのは Torobi の方針(§3.2) |

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

同日のレビュー(コードと突き合わせ)で直したこと:

- mlx-lm は embedding も量子化して配る。Torobi は embedding を bf16 `[vocab, dim]` で宣言し
  (`llama.rb:265`)、tied の head はその表を読む(`llama.rb:257`)ので、(A') の規則無しには
  Phase 1 の実物が開けない。「engine の読み込み経路に変更は要らない」を撤回した。
- メモリピークの原因を「lazy eval」から「`files` の `HashMap` がハンドルを握る寿命」に
  直した(`plan.rs:143-147`、`:373`)。層ごと `eval` では直らない。
- `QuantizedMatmul::vjp` が断るのは weight だけで、scales / biases は微分できる(見込み)。
  3 つとも断るのは Torobi の方針として書き直し、Phase 0 で確かめる行を足した。
- 0.26 GiB は embedding も 4.5 bit にした数字だった。embedding を bf16 に残すと 0.44 GiB。
- Phase 1 の「base は動かない」のテストが `fetch` を使うので、`fetch` の dequantize を
  Phase 3 から Phase 1 に前倒しした。
- `export.rs` が全パラメータを f32 で書くこと、`param_norm` が全パラメータを数えること。
  どちらも packed u32 を数として扱ってしまうので、Phase 1 の拒否 / 除外に入れた。
- §2.3 のシンボル一覧に `affine_quantize` の g64 が無いこと。Phase 0 で確かめる。
- `cast` の検査は `IR::Dtype.check!` が `ALL` を見るので、u32 の明示の除外が要ること。
- `GraphConfig` の検査は「u32 が trainable」の dtype 検査に絞った。DSL では起こり得ない。

## 9. 実装の記録と、まだ言えないこと

実装は plan.md §15.76。この節は「何がまだ確かめられていないか」の一覧で、確かめたら消す。

- **Rust のテストの初回(2026-11-08、GPU 機で `rake rust_test`)**: 204 本中 198 本が通り、
  6 本が落ちた。直したのは (1) `to_tensor` が u32 を断っていたせいで RNG のキー(u32)が
  読めなくなっていた 3 本(断るのをやめ、packed の読み方は `fetch` だけが知る)、
  (2) (B) で量子化した scales が宣言 dtype に変換されずに返っていた 1 本、(3) bf16 の
  `quantized_matmul` の比較基準を bf16 の dequantize → matmul から f32 のそれに変えた
  1 本(重みごとの bf16 丸めが 128 個積もって 2 % になっていた。kernel は f32 で展開し
  出力だけ丸める)、(4) メモリの測定を eval 直後の一発読みからポーリングに変えた 1 本
  (bf16 ぴったり 1 MiB が残って見えた。command buffer の完了ハンドラが入力を離す前に
  読んでいた)。**scales / biases の勾配を MLX が返す**行は通った(§3.2 確定)。
  **直した 6 本を含め、2 回目で 204 本全部が通った。** メモリの件はポーリングで消えた
  ので、握っているものは無い。
- **実物の inventory が無い。** `rake oracle:qwen2_4bit` を Hub に届く機械で一度。
  それまで `test_the_description_declares_what_the_published_4bit_file_holds` は skip する。
- **forward の mlx-lm との比較**(Phase 1 の表の 3 行目)は、手順はあって数字が無い。
  `tools/qlora_reference.py`(`rake oracle:qwen2_4bit_forward`、mlx-lm が要るので Mac か
  CUDA 機)が `test/oracle/qwen2.5-0.5b-instruct-4bit.forward.json` を書き、`test/qlora_test.rb`
  が weights と oracle の揃った機械で bf16 の許容(`Parity::PARITY_BF16 = 1e-2`)で比べる。
  同じテストが、同じファイルを packed のまま読んだ場合と (A') で数値に戻して読んだ場合の
  forward が一致することも見る。oracle が無いうちは skip する。
  **初回の数字(2026-11-08、Mac)**: hidden の最大差は mlx-lm に対して 3.3e-2、Torobi 同士
  (packed と数値読み、bf16)で 4.2e-2(どちらも最大値 207 に対する比)。後者が床で、
  2 つの読み方の違いは重みを bf16 に 1 回丸めるかどうかだけなのにここまで開く。原因は
  Qwen2.5 の残差ストリームに 200 前後の値があり、bf16 の刻みがそこで 1.0 なので、24 層の
  bf16 加算で数刻み分歩くこと。配線の違いなら差は桁で出る。
  **2 回目(MARGIN=1)**: f32 で packed と数値読みを比べると hidden 1.5e-4、logits 9.3e-5
  (`PARITY_PACKED = 1e-3`。両者の違いは f32 の結合順だけ)。**配線は正しい。** bf16 で
  mlx-lm とは、次のトークンが一致し、logits の差 8.2e-3(`PARITY_BF16 = 5e-2`)、hidden は
  3.3e-2 と 5.9e-2(`PARITY_BF16_HIDDEN = 1e-1`。hidden は残差ストリームの外れ値次元の
  ulp の揺れをそのまま受け、logits はその射影で平均される)。上位 10 の集合は bf16 では
  要求しない(丸めで入れ替わり得る)。**3 回目で 3 本とも通った。** Phase 1 の表の
  「forward が mlx-lm と一致する」はこれで実測で埋まり、残りは下の 2 つだけである。
- **Linux の `cargo test --no-run`(link)** もこの箱では未確認。CI の ubuntu が見る。
- **Phase 2 の実測**(7B を RTX 3080 で、32B を Mac 48 GB で開いて `Torobi::Memory` を
  読む)は、手順はあって数字が無い。`tools/qlora_memory.rb <checkpoint-dir>` が bf16 の
  checkpoint(分割されていてよい。engine が `model.safetensors.index.json` を読む)を
  `quantization:` と `adapter:` 付きで開き、batch 1 × 512 token の step を 2 回取って、
  宣言のバイト数と、開いた直後・step 後の active / peak を JSON で出す。`--bf16` で base を
  bf16 のまま開いた数字(表の比較対象)も取れる。`--open-only` は開くだけで止まる。
  §1.2 の表はこれが出るまで見込みのまま。手元の機材は **16 GB の Mac**(表に書いた 48 GB
  ではない)と、今は使えない RTX 3080 なので、順番は Qwen2.5-3B(bf16 5.8 GiB、量子化後
  1.9 GiB 見込み。ピークが抑えられていれば 2.5 GiB 前後、bf16 全体が残れば 6 GiB 前後で、
  どちらでも 16 GB に収まり安全に判別できる)→ 7B(量子化後 4.7 GiB)。32B は 16 GB には
  載らないので、表の 32B の行は機材が揃うまで見込みのまま。
- **PEFT で adapter を読み戻して logits を比べる**(Phase 3 の出口)は未着手。
  `adapter_config.json` の綴りは PEFT の `LoraConfig` の記憶に基づくので、読めなければ
  `Export.adapter_config` を直す。mlx-lm の adapter 形式(`adapters.safetensors`、
  `adapter_config.json` の `lora_parameters`)は別物で、やっていない。
