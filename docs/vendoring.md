# Dependency ledger

What the engine is built against, and how that was decided.

## History

The dependency moved on 2026-09-21. Why it was where it was is kept
below rather than deleted: the older sections are true of the older
arrangement, and the reason they were true is what makes the move's
conditions checkable.

**2026-09-03 - OminiX-MLX adopted.** `mlx-rs` / `mlx-sys` were taken as a
git dependency on `OminiX-ai/OminiX-MLX` (commit `4988a3f`). Upstream's
`mlx-rs` was a generation behind (MLX 0.25 against a 0.30 pre-built
archive), and its `mlx-sys` had no way to build without Xcode: it ran
cmake on its mlx-c from source every time. OminiX's fork tracked MLX 0.32
and fell back to a pre-built MLX when no Metal compiler was present, which
is what let the gem install on a machine without the toolchain.

**2026-09-21 - upstream mlx-rs 0.32, with a system MLX.** The reasons for
the fork are gone. Upstream `mlx-rs 0.32.0` / `mlx-sys 0.6.0` are the same
generation as each other and pin mlx-c `c74db530` over **MLX 0.32.2**, and
`mlx-c` carries `MLX_C_USE_SYSTEM_MLX`, which finds an installed MLX
instead of fetching one. So the pre-built MLX is now handed over through
that option, driven from a generated CMake toolchain file, and the fork is
no longer needed. The cost is the same one the exit always named: it is a
number (the MLX/mlx-c pair), not a fork of `mlx-sys`.

**2026-09-22 - a patch on top, so that a Linux build can exist.** MLX's
CUDA backend is how a graph written on this Mac is trained on a rented
Linux box with an NVIDIA card, and `mlx-rs` could not be built for such a
machine at all: four separate failures, only two of them about CUDA. The
answer is a `[patch.crates-io]` onto a fork, which is a bridge and not a
destination - "The Linux patch, and when it goes" below says what has to
become true for it to be deleted.

The parts below headed "Which one is upstream", "Is the fork's mlx-rs the
same", "What Apple publishes" and "Which mlx-c, and which MLX under it"
are the investigation that led here, kept for its evidence.

## Which mlx-rs

There are three repositories with a claim to the name, and the manifests
do not make it obvious which one is built. In order, top to bottom:

| | what it is | where |
|---|---|---|
| **mlx-rs (upstream)** | the unofficial Rust bindings, by Minghua Wu and David Chavez, and what crates.io publishes. **This is what Torobi builds**, version `0.32.0`, whose `mlx-sys 0.6.0` pins mlx-c and MLX 0.32.2 | `github.com/oxiglade/mlx-rs`, formerly `github.com/oxideai/mlx-rs`. The old name redirects |
| **OminiX-MLX** | **the same repository, continued**, and what Torobi built from 2026-09-03 to 2026-09-21. Not a rewrite and not a vendored copy: the history is mlx-rs's own (534 commits, `init commit` at the bottom, 302 of them by upstream's main author), and on 2026-01-25 `753d289 refactor: Move original mlx-rs components into mlx-rs directory` moved it into a subdirectory to make room for model crates | `github.com/OminiX-ai/OminiX-MLX`, subtree `mlx-rs/` |
| **mlx-c** | Apple's C API for MLX, a git submodule of `mlx-sys`. bindgen reads its headers | `github.com/ml-explore/mlx-c` |
| **MLX** | the library itself. Not built here: a pre-built binary is downloaded at build time (below) | `github.com/ml-explore/mlx` |

## Which one is upstream, and why nothing says so

`oxiglade/mlx-rs` is. GitHub's API settles it rather than the pages do:

| | `fork` | created | stars |
|---|---|---|---|
| `oxiglade/mlx-rs` | **false**, no parent | 2023-12-23 | 371 |
| `OminiX-ai/OminiX-MLX` | **false**, no parent | **2026-01-26** | |

The first is the original: not a fork, created when mlx-rs began, holding
the stars and the docs, and published to crates.io by minghuaw and dcvz,
who are the authors of most of the history in both. `oxideai/mlx-rs`
redirects to it, so that was a rename and not a move to a fork.

The second is not a GitHub fork of anything. It was created on 2026-01-26,
**the day after** `753d289 Move original mlx-rs components into mlx-rs
directory` in its own history: a restructured clone, pushed as a new
repository rather than forked.

**That is why this is confusing, and the reason is worth stating.** The
descent is real in git and invisible to GitHub: no parent link, no fork
banner, and crate metadata that names a third URL. Nothing you can click
on tells you what was compiled. Only the git history and `Cargo.lock` do.

## Is the fork's mlx-rs the same as the one on crates.io

Same origin, still converging, not the same code.

**Still converging**: upstream's work keeps arriving. The history carries
upstream pull requests (#289, #305, #311, #313, #314, #321, #323) and a
commit that says how they get there, `b6c36f3 fix: remove duplicate
gather_mm and Float64 pattern after upstream rebase`. The Rust API is
upstream's, which is what makes the exit below realistic.

**Not the same code**: twelve commits by OminiX-era authors touch
`mlx-rs/` and upstream does not have them. Sorted by what they are, and
by whether Torobi is standing on them:

| | commits | reaches Torobi |
|---|---|---|
| build machinery | `100f155` auto-download pre-built MLX when Xcode is unavailable, `b024efa` deployment-target override | **`100f155` yes, and it is the whole reason this fork is here.** Without it `gem install` needs Xcode |
| the library | `e53aa1b` contiguity check in `try_as_slice` / `contiguous` (a breaking change: a non-contiguous array is an error now rather than the wrong bytes), `c13ee0e` Float64 in safetensors, `284916e` IO extensions, `d145d5b` requires MLX 0.32.0 | `e53aa1b` is in a function the engine calls (`tensor.rs` copies through `contiguous()`), but the engine calls it *to make* the array contiguous, so it stands on its own rather than on the check. The other three it does not use |
| structure | `753d289` the move into a subdirectory, `e5aed65` the version renumbering, `b6c36f3` cleanup after an upstream rebase | no |
| fixes elsewhere | `6d11748` RoPE reshape that broke multi-head attention, `d8495fd` and `e4beb9b` async pipelining | no: the engine implements RoPE itself (`interp.rs`, host-built angles and a rotation), so `nn::Rope` is never called |

So the answer to "could we build against upstream instead" is: the Rust
side, yes, checked function by function; the build without Xcode, no.
That is the exit below, and this is the evidence for how narrow it is.

**The version numbers do not compare.** The fork renumbered to 1.0.0 in
`e5aed65 feat: v1.0.0 - version alignment with OminiX-API` and is 1.2.0
now; crates.io's newest is 0.25.3, and upstream's own README still tells
you to install 0.21.0. Three numbers, no relation between any two of
them. `Cargo.lock`'s commit is the only version that means anything
here.

**The crate metadata points at the wrong one.** OminiX's workspace still
carries upstream's `repository = "https://github.com/oxideai/mlx-rs"`, so
`cargo tree`, docs.rs links and anything reading crate metadata lead to
upstream rather than to what was compiled. Read `engine/Cargo.toml` and
`Cargo.lock` for the truth; they name the fork and its commit.

## What Apple publishes, and why it is not a drop-in

`mlx-metal` on PyPI is official (`mlx@group.apple.com`, owners awni /
katharas / mlx-dev), versioned per MLX release, one wheel per macOS
version, **with a SHA-256 for every file**. At 0.32.2 a wheel holds:

| | |
|---|---|
| `mlx/lib/mlx.metallib` | 129.55 MB, the compiled Metal kernels |
| `mlx/lib/libmlx.dylib` | 20.86 MB, **dynamic** |
| `mlx/lib/libjaccl.dylib` | 1.48 MB |
| `mlx/include/**` | MLX's headers, 405 entries |

What `mlx-sys` wants from `MLX_PREBUILT_PATH` is `libmlx.a`, `libmlxc.a`,
`libgguflib.a` and `mlx.metallib`: **static** libraries, and `libmlxc.a`
is mlx-c, a separate project that is in no MLX wheel. So the official
artifact is not a substitute for OminiX's tarball as things stand.

What it *is* good for is the expensive half: the metallib needs the Metal
compiler and is 130 MB, and Apple publishes it, versioned and hashed.

**But a prebuilt metallib does not buy a build without the toolchain**,
which was worth checking before planning around it. MLX asks for the
Metal compiler twice, and `MLX_METAL_PATH` is not a way past either:

```cmake
# CMakeLists.txt, when MLX_BUILD_METAL=ON on Darwin: fatal at configure
execute_process(COMMAND zsh -c "echo __METAL_VERSION__ | xcrun -sdk macosx metal ..."
                COMMAND_ERROR_IS_FATAL ANY)

# mlx/backend/metal/kernels/CMakeLists.txt: every .metal compiled and linked
add_custom_command(OUTPUT ${MLX_METAL_PATH}/mlx.metallib
                   COMMAND xcrun -sdk macosx metal ...)
add_dependencies(mlx mlx-metallib)   # the mlx target depends on it
```

`MLX_METAL_PATH` moves where the metallib is written and looked for; it
does not skip building it, and `add_subdirectory(kernels)` is
unconditional. So anyone building MLX with Metal needs the toolchain
(`xcodebuild -downloadComponent MetalToolchain`), which is a thing a
GitHub macOS runner can install and this machine cannot.

## Which one to point at

Three different questions, three different answers, and they are not in
conflict:

| asking | look at |
|---|---|
| what does this API do | **upstream**, `oxiglade.github.io/mlx-rs`. The API was always upstream's, and OminiX published no documentation of its own for it |
| what is actually compiled | **the pinned version**, through `engine/Cargo.toml` and `Cargo.lock` |
| what may be redistributed, and on whose terms | **upstream mlx-rs**, which is dual-licensed MIT or Apache-2.0 |

The dependency is the crates.io crate now. It installs without Xcode
because the MLX it is built against is provided as a system package
(below), not because a fork reaches for one.

## The Linux patch, and when it goes

The crates on crates.io cannot be built for anything that is not Apple's.
This matters because MLX itself can: `ml-explore/mlx` carries a CUDA
backend, builds and tests it on every pull request against CUDA 12.6,
12.9 and 13.0, and publishes `mlx-cuda-12` / `mlx-cuda-13` to PyPI in step
with the Metal wheels. The backend is there; the bindings do not reach it.

Four things stop them, and it is worth saying that only two are about
CUDA. The other two would stop a CPU-only Linux build just as dead:

| | where | what |
|---|---|---|
| 1 | `mlx-sys/build.rs` | `dylib=objc` and `framework=Foundation` are linked for every target, cmake is handed `/usr/bin/cc` for the macOS SDK, and `clang_rt.osx` is looked up through `xcrun` |
| 2 | `mlx-sys` | no way to ask for the CUDA backend: `MLX_BUILD_CUDA` is never set, and because MLX is linked as a **static** archive its CUDA dependencies (cuBLASLt, cuFFT, NVRTC, the driver API, cuDNN) arrive at the final link unresolved and must be named there too |
| 3 | `mlx-rs/src/random.rs` | `RandomState::new` seeds itself from `mach_approximate_time`, Mach's clock, reached from a `thread_local!`. Any use of `mlx_rs::random` therefore fails to link off Apple - and the engine uses it in ten places (`init`, dropout in `interp`, `executor`, `plan`, `state`). The patch does not port the clock, it drops it: the value only ever reaches `key()`, so `std::hash::RandomState` answers the same question with no dependency and no `cfg`, and is the better seed besides - a clock is shared by two runs started close enough together, which on one training box is a thing that happens |
| 4 | `mlx-rs/Cargo.toml` | `mlx-sys` is taken with its defaults on, so `default-features = false` on `mlx-rs` turns off this crate's `metal` and leaves the sys crate linking Metal regardless. No feature combination produces a build that is not Apple's |

**The exit condition, which is the point of writing this down.** The
patch is deleted - not edited, deleted - when a published `mlx-sys` and
`mlx-rs` carry these four. Concretely: when a crates.io release builds for
`x86_64-unknown-linux-gnu` and offers a `cuda` feature, `[patch.crates-io]`
comes out of the workspace manifest and `engine/Cargo.toml` goes back to
naming versions alone. Nothing else in the tree depends on the fork, which
is what keeps that a deletion rather than a migration.

Until then the fork is `takahashim/mlx-rs`, branch `linux-cuda`, pinned by
revision rather than by branch: a branch is a name that moves, and what is
compiled here should be a thing that does not. The four commits are one
per row of the table above, and each leaves macOS behaviour alone - which
is checkable, and was checked, by building and testing this project
against the patch on a Mac before anything else was done with it.

This is the second time a fork has been carried here, and the first one
(OminiX-MLX, above) is the reason the exit is written before the entrance.

## The ledger

| what | state |
|---|---|
| mlx-rs / mlx-sys | **crates.io, `mlx-rs = "=0.32.0"` with `mlx-sys = "=0.6.0"`.** `Cargo.lock` records the versions and the checksums. Both are exact while the move settles. Both are also **patched**, by revision, onto `takahashim/mlx-rs` so that a Linux build can exist at all; "The Linux patch, and when it goes" above says what the patch is and what deletes it. The versions are unchanged by it, which is what lets `=0.32.0` and `=0.6.0` keep meaning what they say. `Cargo.lock` shows **four** crates coming from the fork rather than the two named: `mlx-macros` and `mlx-internal-macros` follow because the patched `mlx-rs` takes its workspace siblings by path. Removing the patch returns all four at once |
| mlx-c | the submodule inside `mlx-sys 0.6.0`, pinned there to `c74db530` (v0.6.0-7). bindgen reads its headers; nothing of ours fetches it separately |
| MLX core | **not built from source here**: there is no Metal compiler on this machine, so a pre-built archive is used instead. `takahashim/mlx-prebuilt`, built from stated inputs on a runner with the toolchain rather than taken from a third party's release. It must carry the generation `mlx-sys 0.6.0` was generated for - **MLX 0.32.2 over mlx-c `c74db5307cc8`** - and `mlx_prebuilt.json`'s `requires` is what holds it to that. It names the mlx-c **commit**, not `v0.6.0`: the tag pins MLX 0.31.1 and seven later commits pin 0.32.2, and those commits change headers under `mlx/c/`, so a build from the tag is the wrong generation even though it says 0.6.0. The pin now names **`takahashim/mlx-prebuilt` `v0.6.0.0`** (MLX 0.32.2 / mlx-c `c74db530`, digest `09f634a1…`), which is what a build fetches |
| mlx.metallib | 105 MB. MLX locates it through `dladdr`, i.e. **beside whichever library holds the MLX symbols**: the installed bundle, or `lib/torobi/` for a checkout. It comes from the prefix at `<prefix>/lib/mlx.metallib`; `ext/torobi/extconf.rb` installs it beside the bundle and `rake metallib` copies it into the checkout. Any distribution must ship it beside the bundle |
| system MLX, how it is handed over | `mlx-c` is configured with `MLX_C_USE_SYSTEM_MLX=ON` and `CMAKE_PREFIX_PATH=<prefix>`, through a toolchain file `mlx_prebuilt.rb` writes into the cache and cargo is told about via `CMAKE_TOOLCHAIN_FILE`. MLX is then found, never fetched, and never compiled. The link path upstream does not emit for a system MLX is added with `RUSTFLAGS=-L native=<prefix>/lib` |

## Licences

| | licence | holder |
|---|---|---|
| Torobi | MIT | this project |
| mlx-rs (upstream, oxiglade) | MIT **or** Apache-2.0, at the user's choice | its authors |
| mlx-c | MIT | ml-explore |
| MLX | MIT | ml-explore |

(OminiX-MLX, used from 2026-09-03 to 2026-09-21, was under the same
MIT-or-Apache-2.0 terms; nothing of it is in the current build.)

Everything in the chain is permissive, and MIT and Apache-2.0 both ask the
same thing of a redistributor: carry the notice.

**Torobi carries none of it today, and does not have to.** What the gem
holds is `spec.files`: Ruby, the engine's own Rust, two manifests and the
docs. No line of MLX or mlx-rs is in it. Cargo fetches mlx-rs from
crates.io, and `ext/torobi/mlx_prebuilt.rb` fetches our own pre-built MLX
from `takahashim/mlx-prebuilt`. Both arrive at the user's machine from
their own authors, under their own licences; Torobi points, it does not
ship.

**One decision changes that.** If the distribution question
(docs/plan.md section 11.4) lands on a **platform gem** (compiled, so that
nothing is built at install), then the package contains MLX's compiled
code and its 105 MB `mlx.metallib`, and Torobi becomes a redistributor.
What that costs, exactly:

- ship MLX's MIT notice and copyright (ml-explore)
- ship mlx-c's MIT notice (ml-explore)
- ship mlx-rs's notice under whichever of MIT or Apache-2.0 is chosen
  (MIT is the simpler pairing with this project's own licence)
- say in the README what is inside the binary and under what terms

Not hard, and not something to discover afterwards: it is written here so
that the platform-gem decision is made with it in view.

## Why a pinned dependency rather than a vendored copy

The plan (docs/plan.md section 5) first imagined selective vendoring:
copying array / ops / fast / transforms / io into this tree and pruning the
rest. A pinned dependency does the same job for less:

- it costs no source tree of ours to carry or to re-sync
- cargo records the exact version and checksum in Cargo.lock, so the pin is
  enforced rather than described
- it builds anywhere, which is the whole point (below)

What it gives up is local pruning and local patching. If a patch becomes
necessary, the answer is a fork with its own pin, not a vendored copy.

This was a pinned **git dependency on OminiX's fork** until 2026-09-21,
because that fork tracked MLX 0.32 and reached for a pre-built MLX. Both
of those are now true of the crates.io crate and of mlx-c's own
`MLX_C_USE_SYSTEM_MLX`, so the fork is gone.

## How MLX is provided, and what OminiX did

### Now: a system MLX, found through a toolchain file

Upstream `mlx-sys 0.6.0` always runs cmake on its vendored mlx-c. It has
no branch for "use this MLX already on the machine", but mlx-c does:
`MLX_C_USE_SYSTEM_MLX=ON` turns its `FetchContent(mlx)` into
`find_package(MLX REQUIRED)`, and MLX installs a real CMake package, so
nothing of MLX is fetched or compiled. The `cmake` crate reads
`CMAKE_TOOLCHAIN_FILE` from the environment, which is how those two
variables reach that cmake run without touching `mlx-sys`:

```cmake
set(MLX_C_USE_SYSTEM_MLX ON CACHE BOOL "" FORCE)
set(CMAKE_PREFIX_PATH "<prefix>" CACHE STRING "" FORCE)
set(MLX_C_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
```

The third is not optional. `mlx-c` builds its examples by default, and
`example-gguf` / `example-safe-tensors` link MLX's vendored `gguflib`,
which exists in the build tree only when MLX is built there too; with a
system MLX they fail to link (`_gguf_*` undefined) and stop the build.
The library does not need them.

`ext/torobi/mlx_prebuilt.rb` writes that file (into the cache, so no
machine-specific path is committed) and fetches and checks the prefix that
it names. `extconf.rb` and the Rakefile set `CMAKE_TOOLCHAIN_FILE` for
cargo. Two further variables come from the same place: `MLX_RS_METAL_PATH`
tells `mlx-sys` where the metallib is (the prefix's `lib/`, so it is found
rather than warned about), and `RUSTFLAGS=-L native=<prefix>/lib` adds the
one link path upstream does not emit when MLX is a system package, since
`mlx-sys` looks for `libmlx.a` / `libgguflib.a` in its own build tree.

### Then: OminiX's `build.rs`

Kept because it is what the exit above was measured against.
`mlx-sys/build.rs` decided in three ways:

| condition | what it does |
|---|---|
| `MLX_PREBUILT_PATH` is set | links `libmlx.a` / `libmlxc.a` / `mlx.metallib` from that directory. **This is the branch Torobi takes**, below |
| `xcrun -sdk macosx metal --version` works | builds MLX from source through cmake, patching `device.cpp` / `device.h` to disable NAX for older Metal |
| no Metal compiler (this machine) | downloads `mlx-prebuilt-v0.1.0-macos-arm64.tar.gz` from OminiX's releases |

This machine takes the third: `xcrun -sdk macosx metal --version` answers
`cannot execute tool 'metal' due to missing Metal Toolchain`.

**The download is `curl -L -f` and nothing else.** No checksum, no
signature, nothing in `build.rs` that verifies what arrived beyond TLS
(`grep -n 'sha256|checksum|verify|digest'` finds nothing). It runs during
`gem install`, on the machine of whoever installs, and announces itself
as `cargo:warning=Downloading pre-built MLX from: ...`. Whoever wants
that guarantee sets `MLX_PREBUILT_PATH` to a directory they trust; that
is the first branch, and it exists for exactly this.

**Where it lands, and who moves it afterwards.** `build.rs` extracts into
`OUT_DIR/mlx-prebuilt/` (reused on later builds if the three files are
there) and then copies the metallib to `target/<profile>/mlx.metallib`.
Everything past that point is Torobi's own: `extconf.rb` appends a
Makefile rule that puts it beside the installed bundle, `rake metallib`
puts it in `lib/torobi/` for a checkout, and the command line finds it
beside itself in `target/`. `Torobi::Preflight` and `runtime.rs` refuse
when it is missing, which is the end of the chain and the reason a
missing file is an error rather than an abort.

bindgen runs in every mode, from the mlx-c headers in the submodule.

**Torobi takes the first branch on purpose.** `ext/torobi/mlx_prebuilt.rb`
fetches the archive itself, refuses anything whose SHA-256 is not the
recorded one, unpacks it into `~/.cache/torobi/`, and hands that directory
to cargo through the environment (`extconf.rb` exports it into the
generated Makefile; the Rakefile sets it for the engine's own builds). So
the third branch never runs, one checked copy serves every build on the
machine, and a replaced release asset breaks against the digest instead of
being linked. That digest was computed from the bytes here and agrees with
the one GitHub publishes for the asset.

## Which mlx-c, and which MLX under it

**The generation is now stated by the dependency itself.** `mlx-sys 0.6.0`
pins mlx-c `c74db530` (v0.6.0-7) over **MLX 0.32.2**, its submodule and its
`CHANGELOG.md` both say so, and `ext/torobi/mlx_prebuilt.json`'s `requires`
refuses a pre-built archive that reports anything else. The rest of this
section is the investigation that made that the deciding question; the
`v0.4.1` / `v0.30.1` pair below is what was current before the move.

`mlx-sys` generates its bindings from mlx-c's headers, so what the archive
has to match is mlx-c, not MLX. OminiX's copy of mlx-c differed from
upstream's `v0.4.1` by exactly one line, and it is not a header:

```diff
-    GIT_TAG v0.30.1)
+    GIT_TAG v0.32.0)
```

That line decides which MLX the from-source path compiles; `mlx/c/ops.h`
and the rest of the headers are identical to the tag. So the bindings are
mlx-c v0.4.1's either way, and an archive of **mlx-c v0.4.1 with MLX
v0.30.1** is the pairing upstream tagged and the one those headers were
written against. OminiX's from-source path is the asymmetric one: mlx-c
0.4.1 over an MLX two minor versions newer than it was written for.

This was worth checking rather than assuming, and the assumption made
first here was wrong: an earlier note in this ledger said the headers
expected 0.32.0 and the binary was behind them. The headers expect mlx-c,
and mlx-c is what they got.

Two things follow.

**The unknown MLX revision is findable.** The build script says the
pre-built form "is what ships in the official `mlx-metal` wheel, which is
the only way to obtain a build of MLX's Metal kernels without the Metal
compiler". So the tarball is a repackaged wheel, and the version that
wheel names is the MLX revision this ledger cannot currently state.

**There is an exit, and it is narrow.** What Torobi depends on OminiX for
is not the Rust API - the engine touches about twelve functions
(`matmul`, the four arithmetic ops, `square`, `mean`, `transpose_axes`,
`contiguous`, `stop_gradient`, `value_and_grad_with_argnums`, `eval`,
`Array::from_slice`, `Exception`) - but the build machinery: building
without Xcode. That is replaceable:

1. obtain the official `mlx-metal` wheel (pip download, unpack), and
2. point `MLX_PREBUILT_PATH` at it.

**That exit is narrower than it was written.** Upstream's `mlx-sys` has no
prebuilt branch at all: `build_and_link_mlx_c` runs cmake on its mlx-c
submodule every time, with `MLX_BUILD_METAL=ON` under the `metal` feature,
and MLX refuses to configure that without the Metal toolchain. The
`MLX_RS_METAL_PATH` it does read only chooses where the metallib it builds
is cached; there is nowhere to hand it one.

So `MLX_PREBUILT_PATH` is OminiX's, and it is the whole of what Torobi
cannot get upstream:

| | upstream mlx-rs | OminiX-MLX |
|---|---|---|
| with the Metal toolchain | works, building MLX itself | works |
| without it (this machine, and most installs) | **cannot build** | works |

**But there is a way through that needs no branch and no patch**, found
by reading rather than assuming, and tried rather than argued:

1. `mlx-c` has an option for exactly this. Its CMakeLists says
   `if(MLX_C_USE_SYSTEM_MLX) find_package(MLX REQUIRED) else() FetchContent ...`,
   so a pre-installed MLX means MLX is never fetched and never compiled.
2. MLX's install exports a real package (`MLXTargets.cmake`,
   `MLXConfig.cmake` into `share/cmake/MLX`), so `find_package` has
   something to find.
3. The `cmake` crate reads `CMAKE_TOOLCHAIN_FILE` from the environment
   (`cmake-0.1.58/src/lib.rs:450`), so variables can be put into upstream
   mlx-sys's cmake run the same way `MLX_PREBUILT_PATH` is put into
   OminiX's: an exported path, no fork.

Tried, with crates.io's `mlx-rs = "0.25"` unmodified and a toolchain file
setting `MLX_C_USE_SYSTEM_MLX=ON` and `CMAKE_PREFIX_PATH`. **`metal` is a
default feature of both crates, and MLX was still never built**: cmake
configured, skipped the fetch, and went on to compile mlx-c's own C++
against the MLX it was given. The Metal toolchain was never asked for.

It stopped there, on the version pair rather than the mechanism:

```
mlx-prefix/include/mlx/fast.h:52:  std::optional<array> mask_arr = {},
CMakeFiles/mlxc.dir/mlx/c/fast.cpp.o] Error 1
```

`mlx-sys 0.2.0`'s vendored mlx-c is written against **MLX v0.25.1**, and
it was handed v0.30.1's headers. So the price of the published crate,
today, is running MLX five minor versions back. What the archive must
carry also changes: an install tree (lib, include, `share/cmake/MLX`)
rather than four loose files.

`mlx-prebuilt` installs a prefix rather than gathering four files, so one
archive serves both ways in: the tree is what `find_package(MLX)` follows,
and `lib/` holds what a Rust build links. It is nested under a name
because `lib` is cmake's word for that directory and a bare `lib/` reads
like a gem's; renaming it is not available, since the exported package
records paths relative to the prefix. Torobi searches the unpacked archive
for the prefix rather than knowing where it is, so the shape stays the
archive's business.

**The exit was taken on 2026-09-21**, once upstream had moved to the
0.32 generation and the version pair stopped being the obstacle. Its cost
turned out to be a number (the MLX/mlx-c pair the archive must carry)
rather than a fork or a patch, which is what made taking it cheap. What
remains is the distribution question (docs/plan.md 11.4): a platform gem
builds nothing at install time either way, so the value of any pre-built
path changes with it. The dependency and the distribution question are
still two faces of one decision.

## Updating the pin

Two things move together, because they are one decision: the `mlx-rs` /
`mlx-sys` versions in `engine/Cargo.toml`, and the MLX/mlx-c generation in
`ext/torobi/mlx_prebuilt.json`. One version at a time, as its own change:

1. move `mlx-rs` (and with it `mlx-sys`) to the new version, `cargo update`,
   and read what `mlx-sys`'s `CHANGELOG.md` and submodule now pin;
2. set `requires` in `mlx_prebuilt.json` to that MLX/mlx-c generation;
3. build and release a `takahashim/mlx-prebuilt` archive for it (bump
   `MLX_C_REF` in its `build.sh` to the mlx-c commit `mlx-sys` pins), then
   `rake mlx:pin[v<release>]`;
4. rebuild, and run the differential, convergence, memory and installed-gem
   tests (docs/plan.md section 12).

`Torobi::Native.build_info` reports what a build was made from:
`mlx_rs` and `mlx_sys` are the versions from `engine/Cargo.toml`, read at
build time by `engine/build.rs`.

## What the installed-gem smoke test found

`ruby test/installed_gem_smoke.rb` builds the gem, installs it into an
empty GEM_HOME, and runs one step from outside the checkout.

**2026-09-03, with the path dependency**: the mechanics worked (a 56 KB
source gem built the extension, the metallib landed where dladdr looks, a
step ran), but the gem only built on this machine: `engine/Cargo.toml`
named `/Users/maki/git/OminiX-MLX/mlx-rs` by absolute path, and that path
shipped inside the package. Moving the checkout aside and installing again
failed with `make: *** [target/release/libtorobi.dylib] Error 101`.

**2026-09-03, with the git dependency**: the same test **with the local
checkout renamed away** builds, installs and runs a step (loss 2.5). The
package no longer depends on anything outside itself and its pinned
remotes. That is the finding this ledger existed to reach.
