# Dependency ledger

What the engine is built against, and how that was decided.

## Which mlx-rs

There are three repositories with a claim to the name, and the manifests
do not make it obvious which one is built. In order, top to bottom:

| | what it is | where |
|---|---|---|
| **mlx-rs (upstream)** | the unofficial Rust bindings, by Minghua Wu and David Chavez, and what crates.io publishes (0.25.3, December 2025). **Not what Torobi builds** | `github.com/oxiglade/mlx-rs`, formerly `github.com/oxideai/mlx-rs`. The old name redirects, and both crates.io and OminiX still print it, which is one source of the confusion |
| **OminiX-MLX** | **the same repository, continued.** Not a rewrite and not a vendored copy: the history is mlx-rs's own (534 commits, `init commit` at the bottom, 302 of them by upstream's main author), and on 2026-01-25 `753d289 refactor: Move original mlx-rs components into mlx-rs directory` moved it into a subdirectory to make room for model crates. **This is what Torobi builds**, pinned to one commit | `github.com/OminiX-ai/OminiX-MLX`, subtree `mlx-rs/` |
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
| what does this API do | **upstream**, `oxiglade.github.io/mlx-rs`. The API is upstream's and OminiX publishes no documentation of its own for it |
| what is actually compiled | **the pinned commit**, through `engine/Cargo.toml` and `Cargo.lock`, or the checkout under `~/.cargo/git/checkouts/`. Neither GitHub page will tell you |
| what may be redistributed, and on whose terms | **OminiX-MLX's tree**, which carries upstream's dual licence and the fork's own work under it |

The dependency itself stays where it is. OminiX-MLX is the only one of
the two that installs without Xcode, and that is not a preference.

## The ledger

| what | state |
|---|---|
| mlx-rs / mlx-sys / mlx-c | **git dependency on `https://github.com/OminiX-ai/OminiX-MLX.git`, pinned to `4988a3fcfa48b8cb5d0780a501b92c6a41401523`.** Cargo.lock records the same commit; cargo resolves the mlx-c submodule itself |
| MLX core | **not built from source here**: mlx-sys finds no Metal compiler on this machine and downloads OminiX's pre-built binary `mlx-prebuilt-v0.1.0-macos-arm64.tar.gz`. **It is MLX 0.30.1**, from the version string in its `libmlx.a`; the release says nothing, so this is read off the bytes. Note that the pinned mlx-rs carries `d145d5b feat(qwen3-asr): batched decode, and require MLX 0.32.0` from six months after that tarball was built, and nothing checks the two against each other |
| mlx.metallib | 105 MB. MLX locates it through `dladdr`, i.e. **beside whichever library holds the MLX symbols**: `target/release/` for the CLI, the install directory for the extension. `ext/torobi/extconf.rb` appends a Makefile rule that installs it beside the bundle; `rake metallib` does the same for a checkout. Any distribution must ship it beside the bundle |

## Licences

| | licence | holder |
|---|---|---|
| Torobi | MIT | this project |
| OminiX-MLX (mlx-rs, mlx-sys) | MIT **or** Apache-2.0, at the user's choice. `LICENSE-MIT` and `LICENSE-APACHE` sit at its root | its authors, upstream's included |
| mlx-c | MIT | ml-explore |
| MLX | MIT | ml-explore |

Everything in the chain is permissive, and MIT and Apache-2.0 both ask the
same thing of a redistributor: carry the notice.

**Torobi carries none of it today, and does not have to.** What the gem
holds is `spec.files`: Ruby, the engine's own Rust, two manifests and the
docs. No line of MLX or mlx-rs is in it. Cargo fetches the fork at install
time from its own remote, and `mlx-sys` downloads MLX's pre-built binary
from OminiX's releases. Both arrive at the user's machine from their own
authors, under their own licences; Torobi points, it does not ship.

**One decision changes that.** If the distribution question
(docs/plan.md section 11.4) lands on a **platform gem** (compiled, so that
nothing is built at install), then the package contains MLX's compiled
code and its 105 MB `mlx.metallib`, and Torobi becomes a redistributor.
What that costs, exactly:

- ship MLX's MIT notice and copyright (ml-explore)
- ship mlx-c's MIT notice (ml-explore)
- ship OminiX-MLX's notice under whichever of MIT or Apache-2.0 is chosen
  (MIT is the simpler pairing with this project's own licence)
- say in the README what is inside the binary and under what terms

Not hard, and not something to discover afterwards: it is written here so
that the platform-gem decision is made with it in view.

## Why a git dependency rather than a vendored copy

The plan (docs/plan.md section 5) first imagined selective vendoring:
copying array / ops / fast / transforms / io into this tree and pruning the
rest. A pinned git dependency does the same job for less:

- it costs no source tree of ours to carry or to re-sync
- cargo records the exact commit in Cargo.lock, so the pin is enforced
  rather than described
- it builds anywhere, which is the whole point (below)

What it gives up is local pruning and local patching. If a patch becomes
necessary, the answer is a fork with its own pin, not a vendored copy.

The fork rather than crates.io's mlx-rs 0.25: it tracks MLX 0.32 and falls
back to a pre-built MLX when no Metal compiler is present, which is what
lets the gem build on a machine without Xcode.

## How OminiX builds MLX, and how to leave

`mlx-sys/build.rs` decides in three ways:

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

What that does not fix is where the archive comes from: it is still
OminiX's release of an MLX (0.30.1) two minor versions behind the headers
it is linked against. Moving to a build whose inputs are stated means
changing three constants in that file and nothing else.

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

Then upstream mlx-rs from crates.io would do, at the cost of matching the
wheel's MLX version to what that release's C API expects, which is exactly
the work OminiX is doing for us. `patch_metal_version` matters only on the
build-from-source path, so it does not follow us out.

When to take the exit: if OminiX stops tracking MLX, if a patch of our own
becomes necessary (fork instead), or if the distribution decision
(docs/plan.md 11.4) lands on a platform gem - because then nothing is built
at install time and the value of the auto-download disappears. The
dependency and the distribution question are two faces of one decision.

## Updating the pin

One version at a time, as its own change: move the rev, rebuild, and run
the differential, convergence, memory and installed-gem tests
(docs/plan.md section 12). `Torobi::Native.build_info` reports what a build
was made from; it should learn to report this rev.

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
